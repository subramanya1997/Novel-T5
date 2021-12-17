# NLP-Project
NLP Fall 21 Project

[ Data ](./experiments/data/empatheticdialogues) - Dataset : Empathetic Dialogues
[ Train Novel T5](./experiments/train_novel_T5.py) - Training Novel T5 model
[ Test Novel T5](./experiments/test_novel_T5.py) - Testing Novel T5 model
[ Output ](./Output) - Generated Text 
[ gpt2.zip ](./gpt2.zip) - It is model where picked up most of the base code and modified it to work for T5 (https://github.com/devjwsong/gpt2-dialogue-generation-pytorch)

Setup
Step 1. - pip install -r requirements.txt
Step 2. - python ./experiments/train_novel_t5.py  --batch-size=4 --epochs=50  --multitask1=0.8 --ckpt=yourcheckpoint 
Strp 3. - python ./experiments/test_novel_t5.py --modelckpt=yourcheckpoint

Authors:
Subramanya N - snagabhushan@umass.edu
Shashank Srigiri 
Venkata Bramara Parthiv Dupakuntla