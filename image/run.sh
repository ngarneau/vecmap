pip install -r nvidia_requirements.txt
python -m src.scripts.runner --input_path=/input --output_path=/output --cuda=True
python -m src.scripts.generate_tables --input_path=/input --output_path=/output
