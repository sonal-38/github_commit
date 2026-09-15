"""
Neo4j Knowledge Graph Builder for AI Digital Shadow.

Reads normalized GitHub data from Supabase PostgreSQL and constructs an
idempotent knowledge graph representing entities (Repository, Developer, Commit,
PullRequest, Review, ReviewComment, Issue, IssueComment, File) and their connections.
"""
from typing import Any, Dict, List, Optional
from database.supabase_client import get_supabase_client, SupabaseDatabaseError
from database.neo4j_client import get_neo4j_client, Neo4jClient


class KnowledgeGraphBuilder:
    """
    Builds and maintains the Neo4j Knowledge Graph using normalized entities
    persisted in Supabase PostgreSQL.
    """

    def __init__(self, neo4j_client: Optional[Neo4jClient] = None):
        self.supabase = get_supabase_client()
        self.neo4j = neo4j_client or get_neo4j_client()

    def init_schema(self):
        """
        Create uniqueness constraints in Neo4j for each node label to enforce
        entity integrity and optimize MERGE lookups.
        """
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (r:Repository) REQUIRE r.id IS UNIQUE;",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Developer) REQUIRE d.id IS UNIQUE;",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Commit) REQUIRE c.id IS UNIQUE;",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:PullRequest) REQUIRE p.id IS UNIQUE;",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (rv:Review) REQUIRE rv.id IS UNIQUE;",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (rc:ReviewComment) REQUIRE rc.id IS UNIQUE;",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (i:Issue) REQUIRE i.id IS UNIQUE;",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (ic:IssueComment) REQUIRE ic.id IS UNIQUE;",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (f:File) REQUIRE f.id IS UNIQUE;",
        ]
        for query in constraints:
            try:
                self.neo4j.execute_query(query)
            except Exception:
                # Some Aura versions or tiers handle constraints asynchronously
                pass

    def build_repository_graph(self, owner: str, repo: str) -> Dict[str, Any]:
        """
        Reads normalized repository data from Supabase and populates the Neo4j graph.
        Guarantees idempotency by using MERGE for all nodes and relationships.
        """
        # Ensure Neo4j connectivity
        self.neo4j.verify_connectivity()
        self.init_schema()

        owner_clean = owner.strip()
        repo_clean = repo.strip()
        full_name = f"{owner_clean}/{repo_clean}"

        # 1. Fetch Repository from Supabase
        repos = self.supabase.select("repositories", {"full_name": f"eq.{full_name}"})
        if not repos:
            # Fallback check by name and owner
            repos = self.supabase.select(
                "repositories",
                {"name": f"eq.{repo_clean}", "owner_login": f"eq.{owner_clean}"}
            )

        if not repos:
            raise SupabaseDatabaseError(
                f"Repository '{full_name}' was not found in Supabase. "
                "Please run ingestion first via POST /github/repositories/{owner}/{repo}/ingest",
                status_code=404
            )

        repo_record = repos[0]
        repo_id = repo_record.get("id")

        # 2. Fetch all relational data belonging to this repository from Supabase
        commits = self.supabase.select("commits", {"repository_id": f"eq.{repo_id}"})
        prs = self.supabase.select("pull_requests", {"repository_id": f"eq.{repo_id}"})
        issues = self.supabase.select("issues", {"repository_id": f"eq.{repo_id}"})

        # Fetch all developers from Supabase
        developers_list = self.supabase.select("developers")
        dev_map: Dict[int, Dict[str, Any]] = {d["id"]: d for d in developers_list if "id" in d}

        # PR-dependent records
        pr_id_map: Dict[int, Dict[str, Any]] = {p["id"]: p for p in prs if "id" in p}
        pr_ids = list(pr_id_map.keys())

        reviews = []
        review_comments = []
        changed_files = []

        if pr_ids:
            pr_ids_filter = f"in.({','.join(str(pid) for pid in pr_ids)})"
            reviews = self.supabase.select("reviews", {"pull_request_id": pr_ids_filter})
            review_comments = self.supabase.select("review_comments", {"pull_request_id": pr_ids_filter})
            changed_files = self.supabase.select("changed_files", {"pull_request_id": pr_ids_filter})

        # Issue-dependent records
        issue_id_map: Dict[int, Dict[str, Any]] = {i["id"]: i for i in issues if "id" in i}
        issue_ids = list(issue_id_map.keys())
        issue_comments = []

        if issue_ids:
            issue_ids_filter = f"in.({','.join(str(iid) for iid in issue_ids)})"
            issue_comments = self.supabase.select("issue_comments", {"issue_id": issue_ids_filter})

        # Track count of operations
        nodes_created_or_updated = {
            "repositories": 0,
            "developers": 0,
            "commits": 0,
            "pull_requests": 0,
            "reviews": 0,
            "review_comments": 0,
            "issues": 0,
            "issue_comments": 0,
            "files": 0,
        }
        relationships_count = 0

        # --- A. MERGE Repository Node ---
        repo_cypher = """
        MERGE (r:Repository {id: $id})
        ON CREATE SET r.name = $name, r.owner = $owner, r.html_url = $html_url, r.full_name = $full_name
        ON MATCH SET r.name = $name, r.owner = $owner, r.html_url = $html_url, r.full_name = $full_name
        """
        self.neo4j.execute_query(repo_cypher, {
            "id": full_name,
            "name": repo_record.get("name"),
            "owner": repo_record.get("owner_login"),
            "html_url": repo_record.get("html_url") or f"https://github.com/{full_name}",
            "full_name": full_name,
        })
        nodes_created_or_updated["repositories"] = 1

        # Collect developers actively participating in this repository
        relevant_dev_ids = set()
        for c in commits:
            if c.get("developer_id"):
                relevant_dev_ids.add(c["developer_id"])
        for p in prs:
            if p.get("developer_id"):
                relevant_dev_ids.add(p["developer_id"])
        for r in reviews:
            if r.get("reviewer_id"):
                relevant_dev_ids.add(r["reviewer_id"])
        for rc in review_comments:
            if rc.get("developer_id"):
                relevant_dev_ids.add(rc["developer_id"])
        for i in issues:
            if i.get("developer_id"):
                relevant_dev_ids.add(i["developer_id"])
        for ic in issue_comments:
            if ic.get("developer_id"):
                relevant_dev_ids.add(ic["developer_id"])

        # --- B. MERGE Developer Nodes ---
        dev_cypher = """
        MERGE (d:Developer {id: $id})
        ON CREATE SET d.login = $login, d.name = $name, d.email = $email, d.avatar_url = $avatar_url
        ON MATCH SET d.login = $login, d.name = $name, d.email = $email, d.avatar_url = $avatar_url
        """
        for dev_id in relevant_dev_ids:
            dev = dev_map.get(dev_id)
            if dev and dev.get("github_login"):
                login = dev["github_login"]
                self.neo4j.execute_query(dev_cypher, {
                    "id": login,
                    "login": login,
                    "name": dev.get("name") or login,
                    "email": dev.get("email") or "",
                    "avatar_url": dev.get("avatar_url") or "",
                })
                nodes_created_or_updated["developers"] += 1

        # --- C. MERGE Commits and Relationships ---
        # (Repository)-[:HAS_COMMIT]->(Commit)
        # (Commit)-[:AUTHORED_BY]->(Developer)
        commit_cypher = """
        MATCH (r:Repository {id: $repo_id})
        MERGE (c:Commit {id: $sha})
        ON CREATE SET c.sha = $sha, c.message = $message, c.committed_at = $committed_at, c.html_url = $html_url
        ON MATCH SET c.sha = $sha, c.message = $message, c.committed_at = $committed_at, c.html_url = $html_url
        MERGE (r)-[:HAS_COMMIT]->(c)
        """
        commit_dev_cypher = """
        MATCH (c:Commit {id: $sha})
        MATCH (d:Developer {id: $dev_login})
        MERGE (c)-[:AUTHORED_BY]->(d)
        """
        for c in commits:
            sha = c.get("sha")
            if not sha:
                continue
            self.neo4j.execute_query(commit_cypher, {
                "repo_id": full_name,
                "sha": sha,
                "message": c.get("message") or "",
                "committed_at": c.get("committed_at") or "",
                "html_url": c.get("html_url") or "",
            })
            nodes_created_or_updated["commits"] += 1
            relationships_count += 1  # HAS_COMMIT

            dev = dev_map.get(c.get("developer_id"))
            if dev and dev.get("github_login"):
                self.neo4j.execute_query(commit_dev_cypher, {
                    "sha": sha,
                    "dev_login": dev["github_login"],
                })
                relationships_count += 1  # AUTHORED_BY

        # --- D. MERGE Pull Requests and Relationships ---
        # (Repository)-[:HAS_PR]->(PullRequest)
        # (PullRequest)-[:CREATED_BY]->(Developer)
        pr_cypher = """
        MATCH (r:Repository {id: $repo_id})
        MERGE (p:PullRequest {id: $pr_id})
        ON CREATE SET p.number = $number, p.title = $title, p.body = $body, p.state = $state,
                      p.created_at = $created_at, p.html_url = $html_url
        ON MATCH SET p.number = $number, p.title = $title, p.body = $body, p.state = $state,
                     p.created_at = $created_at, p.html_url = $html_url
        MERGE (r)-[:HAS_PR]->(p)
        """
        pr_dev_cypher = """
        MATCH (p:PullRequest {id: $pr_id})
        MATCH (d:Developer {id: $dev_login})
        MERGE (p)-[:CREATED_BY]->(d)
        """
        for p in prs:
            pr_num = p.get("github_pr_number")
            if pr_num is None:
                continue
            stable_pr_id = f"{full_name}#{pr_num}"
            self.neo4j.execute_query(pr_cypher, {
                "repo_id": full_name,
                "pr_id": stable_pr_id,
                "number": pr_num,
                "title": p.get("title") or "",
                "body": p.get("body") or "",
                "state": p.get("state") or "",
                "created_at": p.get("created_at") or "",
                "html_url": p.get("html_url") or "",
            })
            nodes_created_or_updated["pull_requests"] += 1
            relationships_count += 1  # HAS_PR

            dev = dev_map.get(p.get("developer_id"))
            if dev and dev.get("github_login"):
                self.neo4j.execute_query(pr_dev_cypher, {
                    "pr_id": stable_pr_id,
                    "dev_login": dev["github_login"],
                })
                relationships_count += 1  # CREATED_BY

        # --- E. MERGE Reviews and Relationships ---
        # (PullRequest)-[:HAS_REVIEW]->(Review)
        # (Review)-[:WRITTEN_BY]->(Developer)
        review_cypher = """
        MATCH (p:PullRequest {id: $pr_id})
        MERGE (rv:Review {id: $review_id})
        ON CREATE SET rv.github_review_id = $gh_id, rv.state = $state,
                      rv.submitted_at = $submitted_at, rv.html_url = $html_url
        ON MATCH SET rv.github_review_id = $gh_id, rv.state = $state,
                     rv.submitted_at = $submitted_at, rv.html_url = $html_url
        MERGE (p)-[:HAS_REVIEW]->(rv)
        """
        review_dev_cypher = """
        MATCH (rv:Review {id: $review_id})
        MATCH (d:Developer {id: $dev_login})
        MERGE (rv)-[:WRITTEN_BY]->(d)
        """
        review_id_map: Dict[int, Dict[str, Any]] = {rv["id"]: rv for rv in reviews if "id" in rv}

        for rv in reviews:
            pr_parent = pr_id_map.get(rv.get("pull_request_id"))
            if not pr_parent or pr_parent.get("github_pr_number") is None:
                continue
            pr_stable_id = f"{full_name}#{pr_parent['github_pr_number']}"
            review_stable_id = str(rv.get("github_review_id") or rv.get("id"))

            self.neo4j.execute_query(review_cypher, {
                "pr_id": pr_stable_id,
                "review_id": review_stable_id,
                "gh_id": rv.get("github_review_id") or rv.get("id"),
                "state": rv.get("state") or "",
                "submitted_at": rv.get("submitted_at") or "",
                "html_url": rv.get("html_url") or "",
            })
            nodes_created_or_updated["reviews"] += 1
            relationships_count += 1  # HAS_REVIEW

            reviewer = dev_map.get(rv.get("reviewer_id"))
            if reviewer and reviewer.get("github_login"):
                self.neo4j.execute_query(review_dev_cypher, {
                    "review_id": review_stable_id,
                    "dev_login": reviewer["github_login"],
                })
                relationships_count += 1  # WRITTEN_BY

        # --- F. MERGE Review Comments and Relationships ---
        # (Review)-[:HAS_COMMENT]->(ReviewComment)
        # (ReviewComment)-[:WRITTEN_BY]->(Developer)
        rc_cypher = """
        MATCH (rv:Review {id: $review_id})
        MERGE (rc:ReviewComment {id: $comment_id})
        ON CREATE SET rc.github_comment_id = $gh_id, rc.body = $body, rc.path = $path,
                      rc.line = $line, rc.created_at = $created_at, rc.html_url = $html_url
        ON MATCH SET rc.github_comment_id = $gh_id, rc.body = $body, rc.path = $path,
                     rc.line = $line, rc.created_at = $created_at, rc.html_url = $html_url
        MERGE (rv)-[:HAS_COMMENT]->(rc)
        """
        fallback_review_cypher = """
        MATCH (p:PullRequest {id: $pr_id})
        MERGE (rv:Review {id: $review_id})
        ON CREATE SET rv.github_review_id = $gh_id, rv.state = 'COMMENTED',
                      rv.submitted_at = $submitted_at, rv.html_url = $html_url
        MERGE (p)-[:HAS_REVIEW]->(rv)
        """
        rc_dev_cypher = """
        MATCH (rc:ReviewComment {id: $comment_id})
        MATCH (d:Developer {id: $dev_login})
        MERGE (rc)-[:WRITTEN_BY]->(d)
        """
        for rc in review_comments:
            pr_parent = pr_id_map.get(rc.get("pull_request_id"))
            if not pr_parent or pr_parent.get("github_pr_number") is None:
                continue
            pr_stable_id = f"{full_name}#{pr_parent['github_pr_number']}"
            rc_stable_id = str(rc.get("github_comment_id") or rc.get("id"))

            # Associate comment with a review on this PR
            pr_reviews = [rv for rv in reviews if rv.get("pull_request_id") == rc.get("pull_request_id")]
            matched_review_id = None
            if pr_reviews:
                # If there is a review by the same developer, match it; otherwise use the first review
                matching_dev = [rv for rv in pr_reviews if rv.get("reviewer_id") == rc.get("developer_id")]
                target_rv = matching_dev[0] if matching_dev else pr_reviews[0]
                matched_review_id = str(target_rv.get("github_review_id") or target_rv.get("id"))
            else:
                # If no review record exists for this PR in Supabase, create a review container node
                matched_review_id = f"review-pr-{pr_stable_id}"
                self.neo4j.execute_query(fallback_review_cypher, {
                    "pr_id": pr_stable_id,
                    "review_id": matched_review_id,
                    "gh_id": rc.get("github_comment_id") or rc.get("id"),
                    "submitted_at": rc.get("created_at") or "",
                    "html_url": rc.get("html_url") or "",
                })
                nodes_created_or_updated["reviews"] += 1
                relationships_count += 1  # HAS_REVIEW

            self.neo4j.execute_query(rc_cypher, {
                "review_id": matched_review_id,
                "comment_id": rc_stable_id,
                "gh_id": rc.get("github_comment_id") or rc.get("id"),
                "body": rc.get("body") or "",
                "path": rc.get("path") or "",
                "line": rc.get("line") or 0,
                "created_at": rc.get("created_at") or "",
                "html_url": rc.get("html_url") or "",
            })
            nodes_created_or_updated["review_comments"] += 1
            relationships_count += 1  # HAS_COMMENT

            dev = dev_map.get(rc.get("developer_id"))
            if dev and dev.get("github_login"):
                self.neo4j.execute_query(rc_dev_cypher, {
                    "comment_id": rc_stable_id,
                    "dev_login": dev["github_login"],
                })
                relationships_count += 1  # WRITTEN_BY

        # --- G. MERGE Changed Files and Relationships ---
        # (PullRequest)-[:CHANGED]->(File)
        file_cypher = """
        MATCH (p:PullRequest {id: $pr_id})
        MERGE (f:File {id: $file_id})
        ON CREATE SET f.filename = $filename, f.status = $status, f.additions = $additions,
                      f.deletions = $deletions, f.changes = $changes
        ON MATCH SET f.filename = $filename, f.status = $status, f.additions = $additions,
                     f.deletions = $deletions, f.changes = $changes
        MERGE (p)-[:CHANGED]->(f)
        """
        seen_files = set()
        for f in changed_files:
            pr_parent = pr_id_map.get(f.get("pull_request_id"))
            if not pr_parent or pr_parent.get("github_pr_number") is None:
                continue
            filename = f.get("filename")
            if not filename:
                continue
            pr_stable_id = f"{full_name}#{pr_parent['github_pr_number']}"
            file_stable_id = f"{full_name}:{filename}"

            self.neo4j.execute_query(file_cypher, {
                "pr_id": pr_stable_id,
                "file_id": file_stable_id,
                "filename": filename,
                "status": f.get("status") or "modified",
                "additions": f.get("additions") or 0,
                "deletions": f.get("deletions") or 0,
                "changes": f.get("changes") or 0,
            })
            if file_stable_id not in seen_files:
                nodes_created_or_updated["files"] += 1
                seen_files.add(file_stable_id)
            relationships_count += 1  # CHANGED

        # --- H. MERGE Issues and Relationships ---
        # (Repository)-[:HAS_ISSUE]->(Issue)
        # (Issue)-[:CREATED_BY]->(Developer)
        issue_cypher = """
        MATCH (r:Repository {id: $repo_id})
        MERGE (i:Issue {id: $issue_id})
        ON CREATE SET i.number = $number, i.title = $title, i.body = $body, i.state = $state,
                      i.created_at = $created_at, i.html_url = $html_url
        ON MATCH SET i.number = $number, i.title = $title, i.body = $body, i.state = $state,
                     i.created_at = $created_at, i.html_url = $html_url
        MERGE (r)-[:HAS_ISSUE]->(i)
        """
        issue_dev_cypher = """
        MATCH (i:Issue {id: $issue_id})
        MATCH (d:Developer {id: $dev_login})
        MERGE (i)-[:CREATED_BY]->(d)
        """
        for i in issues:
            issue_num = i.get("github_issue_number")
            if issue_num is None:
                continue
            stable_issue_id = f"{full_name}#{issue_num}"
            self.neo4j.execute_query(issue_cypher, {
                "repo_id": full_name,
                "issue_id": stable_issue_id,
                "number": issue_num,
                "title": i.get("title") or "",
                "body": i.get("body") or "",
                "state": i.get("state") or "",
                "created_at": i.get("created_at") or "",
                "html_url": i.get("html_url") or "",
            })
            nodes_created_or_updated["issues"] += 1
            relationships_count += 1  # HAS_ISSUE

            dev = dev_map.get(i.get("developer_id"))
            if dev and dev.get("github_login"):
                self.neo4j.execute_query(issue_dev_cypher, {
                    "issue_id": stable_issue_id,
                    "dev_login": dev["github_login"],
                })
                relationships_count += 1  # CREATED_BY

        # --- I. MERGE Issue Comments and Relationships ---
        # (Issue)-[:HAS_COMMENT]->(IssueComment)
        # (IssueComment)-[:WRITTEN_BY]->(Developer)
        ic_cypher = """
        MATCH (i:Issue {id: $issue_id})
        MERGE (ic:IssueComment {id: $comment_id})
        ON CREATE SET ic.github_comment_id = $gh_id, ic.body = $body,
                      ic.created_at = $created_at, ic.html_url = $html_url
        ON MATCH SET ic.github_comment_id = $gh_id, ic.body = $body,
                     ic.created_at = $created_at, ic.html_url = $html_url
        MERGE (i)-[:HAS_COMMENT]->(ic)
        """
        ic_dev_cypher = """
        MATCH (ic:IssueComment {id: $comment_id})
        MATCH (d:Developer {id: $dev_login})
        MERGE (ic)-[:WRITTEN_BY]->(d)
        """
        for ic in issue_comments:
            issue_parent = issue_id_map.get(ic.get("issue_id"))
            if not issue_parent or issue_parent.get("github_issue_number") is None:
                continue
            issue_stable_id = f"{full_name}#{issue_parent['github_issue_number']}"
            ic_stable_id = str(ic.get("github_comment_id") or ic.get("id"))

            self.neo4j.execute_query(ic_cypher, {
                "issue_id": issue_stable_id,
                "comment_id": ic_stable_id,
                "gh_id": ic.get("github_comment_id") or ic.get("id"),
                "body": ic.get("body") or "",
                "created_at": ic.get("created_at") or "",
                "html_url": ic.get("html_url") or "",
            })
            nodes_created_or_updated["issue_comments"] += 1
            relationships_count += 1  # HAS_COMMENT

            dev = dev_map.get(ic.get("developer_id"))
            if dev and dev.get("github_login"):
                self.neo4j.execute_query(ic_dev_cypher, {
                    "comment_id": ic_stable_id,
                    "dev_login": dev["github_login"],
                })
                relationships_count += 1  # WRITTEN_BY

        return {
            "repository": repo_clean,
            "full_name": full_name,
            "graph_build_status": "success",
            "nodes_created_or_updated": nodes_created_or_updated,
            "relationships_created_or_updated": relationships_count,
        }

    def get_repository_summary(self, owner: str, repo: str) -> Dict[str, Any]:
        """
        Queries Neo4j for actual entity node counts and relationship count
        connected to this repository.
        """
        self.neo4j.verify_connectivity()
        full_name = f"{owner.strip()}/{repo.strip()}"

        # Verify repository exists in Neo4j
        check_repo = self.neo4j.execute_query(
            "MATCH (r:Repository {id: $repo_id}) RETURN r.name as name",
            {"repo_id": full_name}
        )
        if not check_repo:
            raise SupabaseDatabaseError(
                f"Repository '{full_name}' not found in Neo4j Knowledge Graph. "
                f"Please build the graph first via POST /graph/repositories/{owner}/{repo}/build",
                status_code=404
            )

        summary_query = """
        MATCH (r:Repository {id: $repo_id})
        OPTIONAL MATCH (r)-[:HAS_COMMIT]->(c:Commit)
        OPTIONAL MATCH (r)-[:HAS_PR]->(p:PullRequest)
        OPTIONAL MATCH (p)-[:HAS_REVIEW]->(rv:Review)
        OPTIONAL MATCH (rv)-[:HAS_COMMENT]->(rc:ReviewComment)
        OPTIONAL MATCH (p)-[:CHANGED]->(f:File)
        OPTIONAL MATCH (r)-[:HAS_ISSUE]->(i:Issue)
        OPTIONAL MATCH (i)-[:HAS_COMMENT]->(ic:IssueComment)
        RETURN count(DISTINCT c) as commits,
               count(DISTINCT p) as pull_requests,
               count(DISTINCT rv) as reviews,
               count(DISTINCT rc) as review_comments,
               count(DISTINCT f) as files,
               count(DISTINCT i) as issues,
               count(DISTINCT ic) as issue_comments
        """
        results = self.neo4j.execute_query(summary_query, {"repo_id": full_name})
        counts = results[0] if results else {}

        # Query developers connected to this repository subgraph
        dev_query = """
        MATCH (r:Repository {id: $repo_id})-[*1..3]->(d:Developer)
        RETURN count(DISTINCT d) as developers
        """
        dev_results = self.neo4j.execute_query(dev_query, {"repo_id": full_name})
        dev_count = dev_results[0].get("developers", 0) if dev_results else 0

        # Query total relationships connected in this repository subgraph
        rel_query = """
        MATCH (r:Repository {id: $repo_id})-[rel*1..3]-(n)
        RETURN count(DISTINCT last(rel)) as count
        """
        rel_results = self.neo4j.execute_query(rel_query, {"repo_id": full_name})
        total_rels = rel_results[0].get("count", 0) if rel_results else 0

        return {
            "repository": repo.strip(),
            "full_name": full_name,
            "status": "active",
            "nodes": {
                "repositories": 1,
                "developers": dev_count,
                "commits": counts.get("commits", 0),
                "pull_requests": counts.get("pull_requests", 0),
                "reviews": counts.get("reviews", 0),
                "review_comments": counts.get("review_comments", 0),
                "issues": counts.get("issues", 0),
                "issue_comments": counts.get("issue_comments", 0),
                "files": counts.get("files", 0),
            },
            "relationships_count": total_rels,
        }
