check_dirs := vit_vqgan

style:
	black --preview -l 119 $(check_dirs)
	isort $(check_dirs)