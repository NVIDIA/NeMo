import os

import requests
from github import Github


def send_slack_notification():
    # Get environment variables
    gh_token = os.environ.get('GH_TOKEN')
    webhook_url = os.environ.get('SLACK_WEBHOOK')
    repository = os.environ.get('REPOSITORY')
    run_id = os.environ.get('RUN_ID')
    server_url = os.environ.get('SERVER_URL', 'https://github.com')
    pr_number = int(os.environ.get('PR_NUMBER'))

    # Get failure info from GitHub API
    gh = Github(gh_token)
    repo = gh.get_repo(repository)
    pr = repo.get_pull(pr_number)

    # Get failed jobs
    failed_jobs = [job.name for job in repo.get_workflow_run(int(run_id)).jobs() if job.conclusion == 'failure']

    # Build message blocks
    blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*<{server_url}/{repository}/pull/{pr_number}|PR#{pr_number}: {pr.title.replace('`', '')}>*\n"
                    f"• Author: <{server_url}/{pr.user.login}|{pr.user.login}>\n"
                    f"• Branch: `{pr.head.ref}`\n"
                    f"• Pipeline: <{server_url}/{repository}/actions/runs/{run_id}|View Run>\n"
                    f"• Failed Jobs:\n"
                    + "\n".join(
                        [f"    • <{server_url}/{repository}/actions/runs/{run_id}|{job}>" for job in failed_jobs]
                    )
                ),
            },
        }
    ]

    # Send to Slack
    response = requests.post(webhook_url, json={"blocks": blocks})
    response.raise_for_status()


if __name__ == "__main__":
    send_slack_notification()
