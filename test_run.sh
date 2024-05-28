source prepare_path.sh
python3 scripts/cli.py run -g mm_mapworld -m llava-v1.6-vicuna-13b-hf -i test
python3 scripts/cli.py score -g mm_mapworld
python3 scripts/cli.py transcribe -g mm_mapworld
python3 scripts/cli.py run -g mm_mapworld_graphs -m llava-v1.6-vicuna-13b-hf -i test
python3 scripts/cli.py score -g mm_mapworld_graphs
python3 scripts/cli.py transcribe -g mm_mapworld_graphs
python3 scripts/cli.py run -g mm_mapworld_qa -m llava-v1.6-vicuna-13b-hf -i test
python3 scripts/cli.py score -g mm_mapworld_qa
python3 scripts/cli.py transcribe -g mm_mapworld_qa
python3 scripts/cli.py run -g mm_mapworld_specificroom -m llava-v1.6-vicuna-13b-hf -i test
python3 scripts/cli.py score -g mm_mapworld_specificroom
python3 scripts/cli.py transcribe -g mm_mapworld_specificroom