import os
from datetime import datetime, timedelta, timezone

import gitlab

# Get environment variables
PROJECT_ID = int(os.getenv("CI_PROJECT_ID", 19378))
GITLAB_ENDPOINT = os.getenv('GITLAB_ENDPOINT')
RO_API_TOKEN = os.getenv("RO_API_TOKEN")


def get_gitlab_handle():
    """Initialize and return GitLab client."""
    if not GITLAB_ENDPOINT or not RO_API_TOKEN:
        raise ValueError("GITLAB_ENDPOINT and RO_API_TOKEN must be set")
    return gitlab.Gitlab(f"https://{GITLAB_ENDPOINT}", private_token=RO_API_TOKEN)


def get_most_recent_milestone(project):
    """Get the most recent milestone from the project."""
    milestones = project.milestones.list(state='active', sort='due_date_desc')
    if not milestones:
        return None
    return milestones[0]


def get_age_category(updated_at):
    """Categorize MRs by their age."""
    now = datetime.now(timezone.utc)  # Make now timezone-aware
    updated = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
    age = now - updated

    if age <= timedelta(days=1):
        return "Last 24 hours"
    elif age <= timedelta(days=3):
        return "Last 3 days"
    else:
        return "Older than 3 days"


def main():
    try:
        gl = get_gitlab_handle()
        project = gl.projects.get(PROJECT_ID)

        # Get the most recent milestone
        milestone = get_most_recent_milestone(project)
        if not milestone:
            print("No active milestones found")
            return

        # Fetch all merge requests
        mrs = project.mergerequests.list(
            state='opened',  # Only get open MRs
            labels=['Final Review'],  # Filter by label
            milestone=milestone.title,  # Filter by most recent milestone
            order_by='updated_at',  # Order by update date
            sort='desc',  # Most recent first
        )

        # Group MRs by age
        age_groups = {"Last 24 hours": [], "Last 3 days": [], "Older than 3 days": []}

        for mr in mrs:
            age_category = get_age_category(mr.updated_at)
            age_groups[age_category].append(mr)

        # Print MR details grouped by age
        print(f"Merge requests with 'Final review' label in milestone '{milestone.title}':")
        print("-" * 80)

        for age_group, mrs_in_group in age_groups.items():
            if mrs_in_group:
                print(f"\n{age_group} ({len(mrs_in_group)} MRs):")
                print("-" * 40)
                for mr in mrs_in_group:
                    print(f"**MR**: #{mr.iid} - {mr.title}")
                    print(f"**Author**: {mr.author['name']}")
                    print(f"**Final Review added**: {mr.updated_at}")
                    print(f"**URL**: {mr.web_url}")
                    print("-" * 40)

    except gitlab.exceptions.GitlabAuthenticationError:
        print("Error: Authentication failed. Please check your RO_API_TOKEN")
    except gitlab.exceptions.GitlabGetError as e:
        print(f"Error: Failed to fetch data from GitLab: {e}")
    except Exception as e:
        print(f"Error: An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
