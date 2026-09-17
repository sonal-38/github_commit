"""
Database storage service for persisting normalized GitHub entities into Supabase PostgreSQL.

Implements the relational insertion hierarchy:
1. Repositories
2. Developers
3. Commits
4. Pull Requests
5. Reviews
6. Review Comments
7. Issues
8. Issue Comments
9. Changed Files

Ensures idempotency and duplicate prevention via PostgreSQL upsert constraints.
"""
from typing import Dict, List, Optional, Tuple
from ingestion.normalizer import (
    NormalizedRepository,
    NormalizedCommit,
    NormalizedPullRequest,
    NormalizedReview,
    NormalizedReviewComment,
    NormalizedIssue,
    NormalizedIssueComment,
    NormalizedChangedFile,
    NormalizedCommitFile,
)
from database.supabase_client import SupabaseClient, get_supabase_client, SupabaseDatabaseError


class SupabaseStorageService:
    """Handles structured relational persistence into Supabase tables."""

    def __init__(self, client: Optional[SupabaseClient] = None):
        self.client = client or get_supabase_client()

    def store_all(
        self,
        repository: NormalizedRepository,
        commits: List[NormalizedCommit],
        pull_requests: List[NormalizedPullRequest],
        reviews: List[NormalizedReview],
        review_comments: List[NormalizedReviewComment],
        issues: List[NormalizedIssue],
        issue_comments: List[NormalizedIssueComment],
        changed_files: List[NormalizedChangedFile],
        commit_files: Optional[List[NormalizedCommitFile]] = None,
    ) -> Dict[str, int]:
        """
        Persist all normalized entities into Supabase following foreign-key order.

        Returns:
            Dictionary of stored entity counts.
        """
        # 1. Upsert Repository
        repo_record = {
            "github_id": repository.repository_id,
            "name": repository.name,
            "full_name": repository.full_name,
            "owner_login": repository.owner_login,
            "html_url": repository.html_url,
        }
        repo_res = self.client.upsert("repositories", repo_record, on_conflict="full_name")
        if not repo_res:
            # Fetch existing repository ID if upsert didn't return
            existing = self.client.select("repositories", {"full_name": f"eq.{repository.full_name}"})
            if not existing:
                raise SupabaseDatabaseError(f"Failed to persist repository {repository.full_name}")
            repo_id = existing[0]["id"]
        else:
            repo_id = repo_res[0]["id"]

        # 2. Extract, Upsert Developers & Build Developer ID Map
        developer_dict: Dict[str, Dict[str, Optional[str]]] = {}

        # From Commits
        for c in commits:
            if c.author_login and c.author_login not in developer_dict:
                developer_dict[c.author_login] = {
                    "github_login": c.author_login,
                    "name": c.author_name,
                    "email": c.author_email,
                }

        # From Pull Requests
        for pr in pull_requests:
            if pr.author_login and pr.author_login not in developer_dict:
                developer_dict[pr.author_login] = {
                    "github_login": pr.author_login,
                    "name": None,
                    "email": None,
                }

        # From Reviews
        for r in reviews:
            if r.reviewer_login and r.reviewer_login not in developer_dict:
                developer_dict[r.reviewer_login] = {
                    "github_login": r.reviewer_login,
                    "name": None,
                    "email": None,
                }

        # From Review Comments
        for rc in review_comments:
            if rc.commenter_login and rc.commenter_login not in developer_dict:
                developer_dict[rc.commenter_login] = {
                    "github_login": rc.commenter_login,
                    "name": None,
                    "email": None,
                }

        # From Issues
        for iss in issues:
            if iss.author_login and iss.author_login not in developer_dict:
                developer_dict[iss.author_login] = {
                    "github_login": iss.author_login,
                    "name": None,
                    "email": None,
                }

        # From Issue Comments
        for ic in issue_comments:
            if ic.commenter_login and ic.commenter_login not in developer_dict:
                developer_dict[ic.commenter_login] = {
                    "github_login": ic.commenter_login,
                    "name": None,
                    "email": None,
                }

        dev_id_map: Dict[str, int] = {}
        if developer_dict:
            dev_payload = list(developer_dict.values())
            dev_res = self.client.upsert("developers", dev_payload, on_conflict="github_login")
            for dev in dev_res:
                dev_id_map[dev["github_login"]] = dev["id"]

            # If any were not returned in representation, select them
            missing_logins = [login for login in developer_dict if login not in dev_id_map]
            if missing_logins:
                existing_devs = self.client.select("developers")
                for dev in existing_devs:
                    dev_id_map[dev["github_login"]] = dev["id"]

        # 3. Upsert Commits
        stored_commits = 0
        if commits:
            commit_payload = []
            for c in commits:
                dev_id = dev_id_map.get(c.author_login) if c.author_login else None
                commit_payload.append({
                    "sha": c.sha,
                    "repository_id": repo_id,
                    "developer_id": dev_id,
                    "message": c.message,
                    "committed_at": c.date,
                    "html_url": c.url,
                })
            # Batch upsert
            commit_res = self.client.upsert("commits", commit_payload, on_conflict="repository_id,sha")
            stored_commits = len(commit_res) if commit_res else len(commit_payload)

        # 4. Upsert Pull Requests & Build PR ID Map
        pr_id_map: Dict[int, int] = {}
        stored_prs = 0
        if pull_requests:
            pr_payload = []
            for pr in pull_requests:
                dev_id = dev_id_map.get(pr.author_login) if pr.author_login else None
                pr_payload.append({
                    "github_pr_number": pr.number,
                    "repository_id": repo_id,
                    "developer_id": dev_id,
                    "title": pr.title,
                    "body": pr.body,
                    "state": pr.state,
                    "created_at": pr.created_at,
                    "updated_at": pr.updated_at,
                    "closed_at": pr.closed_at,
                    "merged_at": pr.merged_at,
                    "merge_commit_sha": pr.merge_commit_sha,
                    "html_url": pr.url,
                })
            pr_res = self.client.upsert("pull_requests", pr_payload, on_conflict="repository_id,github_pr_number")
            for item in pr_res:
                pr_id_map[item["github_pr_number"]] = item["id"]
            stored_prs = len(pr_payload)

            # Check if any PR IDs need lookup
            missing_prs = [pr.number for pr in pull_requests if pr.number not in pr_id_map]
            if missing_prs:
                existing_prs = self.client.select("pull_requests", {"repository_id": f"eq.{repo_id}"})
                for pr in existing_prs:
                    pr_id_map[pr["github_pr_number"]] = pr["id"]

        # 5. Upsert Reviews (linked to pull_request_id and reviewer_id)
        stored_reviews = 0
        if reviews:
            review_payload = []
            for r in reviews:
                pr_id = pr_id_map.get(r.pull_request_number)
                if not pr_id:
                    continue
                reviewer_id = dev_id_map.get(r.reviewer_login) if r.reviewer_login else None
                review_payload.append({
                    "github_review_id": r.id,
                    "pull_request_id": pr_id,
                    "reviewer_id": reviewer_id,
                    "body": r.body,
                    "state": r.state,
                    "submitted_at": r.submitted_at,
                    "commit_id": r.commit_id,
                    "html_url": r.url,
                })
            if review_payload:
                review_res = self.client.upsert("reviews", review_payload, on_conflict="github_review_id")
                stored_reviews = len(review_res) if review_res else len(review_payload)

        # 6. Upsert Review Comments (linked to pull_request_id and developer_id)
        stored_review_comments = 0
        if review_comments:
            rc_payload = []
            for rc in review_comments:
                pr_id = pr_id_map.get(rc.pull_request_number)
                if not pr_id:
                    continue
                dev_id = dev_id_map.get(rc.commenter_login) if rc.commenter_login else None
                rc_payload.append({
                    "github_comment_id": rc.id,
                    "pull_request_id": pr_id,
                    "developer_id": dev_id,
                    "body": rc.body,
                    "path": rc.path,
                    "line": rc.line,
                    "diff_hunk": rc.diff_hunk,
                    "commit_sha": rc.commit_id,
                    "created_at": rc.created_at,
                    "updated_at": rc.updated_at,
                    "html_url": rc.url,
                })
            if rc_payload:
                rc_res = self.client.upsert("review_comments", rc_payload, on_conflict="github_comment_id")
                stored_review_comments = len(rc_res) if rc_res else len(rc_payload)

        # 7. Upsert Issues & Build Issue ID Map
        issue_id_map: Dict[int, int] = {}
        stored_issues = 0
        if issues:
            issue_payload = []
            for iss in issues:
                dev_id = dev_id_map.get(iss.author_login) if iss.author_login else None
                issue_payload.append({
                    "github_issue_number": iss.number,
                    "repository_id": repo_id,
                    "developer_id": dev_id,
                    "title": iss.title,
                    "body": iss.body,
                    "state": iss.state,
                    "created_at": iss.created_at,
                    "updated_at": iss.updated_at,
                    "closed_at": iss.closed_at,
                    "html_url": iss.url,
                })
            iss_res = self.client.upsert("issues", issue_payload, on_conflict="repository_id,github_issue_number")
            for item in iss_res:
                issue_id_map[item["github_issue_number"]] = item["id"]
            stored_issues = len(issue_payload)

            missing_issues = [iss.number for iss in issues if iss.number not in issue_id_map]
            if missing_issues:
                existing_issues = self.client.select("issues", {"repository_id": f"eq.{repo_id}"})
                for item in existing_issues:
                    issue_id_map[item["github_issue_number"]] = item["id"]

        # 8. Upsert Issue Comments (linked to issue_id and developer_id)
        stored_issue_comments = 0
        if issue_comments:
            ic_payload = []
            for ic in issue_comments:
                iss_id = issue_id_map.get(ic.issue_number)
                if not iss_id:
                    continue
                dev_id = dev_id_map.get(ic.commenter_login) if ic.commenter_login else None
                ic_payload.append({
                    "github_comment_id": ic.id,
                    "issue_id": iss_id,
                    "developer_id": dev_id,
                    "body": ic.body,
                    "created_at": ic.created_at,
                    "updated_at": ic.updated_at,
                    "html_url": ic.url,
                })
            if ic_payload:
                ic_res = self.client.upsert("issue_comments", ic_payload, on_conflict="github_comment_id")
                stored_issue_comments = len(ic_res) if ic_res else len(ic_payload)

        # 9. Upsert Changed Files (linked to pull_request_id)
        stored_changed_files = 0
        if changed_files:
            cf_payload = []
            for cf in changed_files:
                pr_id = pr_id_map.get(cf.pull_request_number)
                if not pr_id:
                    continue
                cf_payload.append({
                    "pull_request_id": pr_id,
                    "filename": cf.filename,
                    "status": cf.status,
                    "additions": cf.additions,
                    "deletions": cf.deletions,
                    "changes": cf.changes,
                    "sha": cf.sha,
                    "patch": cf.patch,
                    "blob_url": cf.blob_url,
                })
            if cf_payload:
                cf_res = self.client.upsert("changed_files", cf_payload, on_conflict="pull_request_id,filename")
                stored_changed_files = len(cf_res) if cf_res else len(cf_payload)

        # 10. Upsert Commit Files (linked to repository, commit_sha, and filename)
        stored_commit_files = 0
        if commit_files:
            commit_files_payload = []
            for cf in commit_files:
                if not cf.filename or not cf.commit_sha:
                    continue
                commit_files_payload.append({
                    "repository": repository.full_name,
                    "commit_sha": cf.commit_sha,
                    "filename": cf.filename,
                    "status": cf.status,
                    "additions": cf.additions,
                    "deletions": cf.deletions,
                    "changes": cf.changes,
                    "patch": cf.patch,
                    "blob_url": cf.blob_url,
                    "raw_url": cf.raw_url,
                })
            if commit_files_payload:
                chunk_size = 100
                for i in range(0, len(commit_files_payload), chunk_size):
                    chunk = commit_files_payload[i:i + chunk_size]
                    cf_res = self.client.upsert(
                        "commit_files",
                        chunk,
                        on_conflict="repository,commit_sha,filename",
                    )
                    stored_commit_files += len(cf_res) if cf_res else len(chunk)

        return {
            "repositories": 1,
            "developers": len(dev_id_map),
            "commits": stored_commits,
            "pull_requests": stored_prs,
            "reviews": stored_reviews,
            "review_comments": stored_review_comments,
            "issues": stored_issues,
            "issue_comments": stored_issue_comments,
            "changed_files": stored_changed_files,
            "commit_files": stored_commit_files,
        }
