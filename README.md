# NLP-Project
NLP Fall 21 Project

[ Data ](./experiments/data/empatheticdialogues) - Dataset : Empathetic Dialogues <br />
[ Train Novel T5](./experiments/train_novel_T5.py) - Training Novel T5 model <br />
[ Test Novel T5](./experiments/test_novel_T5.py) - Testing Novel T5 model <br />
[ Output ](./Output) - Generated Text <br />
[ gpt2.zip ](./gpt2.zip) - It is model where picked up most of the base code and modified it to work for T5 (https://github.com/devjwsong/gpt2-dialogue-generation-pytorch) <br />
<br />
Setup <br />
Step 1. - pip install -r requirements.txt <br />
Step 2. - python ./experiments/train_novel_t5.py  --batch-size=4 --epochs=50  --multitask1=0.8 --ckpt=yourcheckpoint  <br />
Strp 3. - python ./experiments/test_novel_t5.py --modelckpt=yourcheckpoint <br />

Authors:
Subramanya N - snagabhushan@umass.edu <br />
Shashank Srigiri <br />
Venkata Bramara Parthiv Dupakuntla <br />
