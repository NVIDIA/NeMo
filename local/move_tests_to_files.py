import ruamel.yaml


def main():
    yaml = ruamel.yaml.YAML()

    with open(".github/workflows/cicd-main.yml", "rb") as fh:
        cicd_main = yaml.load(fh)

    for job_name, job in cicd_main["jobs"].items():
        if "with" in job and "SCRIPT" in job["with"]:
            with open(f"tests/functional_tests/{job_name}.sh", "w") as fh:
                fh.write(job["with"]["SCRIPT"])
            cicd_main["jobs"][job_name]["with"]["SCRIPT"] = f"bash tests/functional_tests/{job_name}.sh"

    with open(".github/workflows/cicd-main.yml", "w") as fh:
        yaml.dump(cicd_main, fh)


if __name__ == "__main__":
    main()
