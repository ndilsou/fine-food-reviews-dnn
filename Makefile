
.DEFAULT_GOAL := help
.PHONY: help build push bumpversion

#########################################################
# VARIABLES

project-root=$(shell git rev-parse --show-toplevel)
bucket=machine-learning-store
project-name=amzn-fine-food-reviews

#########################################################
# HELP

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'


#########################################################
# COMMANDS

lab: ## Launch Jupyter lab
	pipenv run jupyter lab --notebook-dir $(CURDIR)/notebooks

upload-%: ## Sync the files of a specific folder with S3.
	aws --profile perso s3 sync \
		--exclude */logs/* \
		--exclude .gitignore \
		$* s3://$(bucket)/$(project-name)/$*

upload: upload-data upload-artefacts upload-reports ## upload the files not tracked in git with S3.
