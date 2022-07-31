check_dirs := vit_vqgan training

quality:
	black -l 119 --check --preview  $(check_dirs)
	isort --check-only $(check_dirs)
	flake8 $(check_dirs)

style:
	black --preview -l 119 $(check_dirs)
	isort $(check_dirs)