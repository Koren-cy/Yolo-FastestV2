import os
from io import TextIOWrapper

script_path = os.path.abspath(__file__)
train_path = os.path.join(os.path.dirname(script_path), "train")
val_path = os.path.join(os.path.dirname(script_path), "val")

train_txt = open(train_path + ".txt", "w", encoding="utf-8")
val_txt = open(val_path + ".txt", "w", encoding="utf-8")   

def process(path:str,txt:TextIOWrapper):
    for file in os.listdir(path):

        file_path = os.path.join(path, file)

        if file.endswith(".jpg"):
            txt.write(file_path + "\n")

        elif file.endswith(".txt"):
            with open(file_path,"r+",encoding="utf-8") as f:
                lines = f.readlines()
                f.seek(0)
                f.truncate()
                for line in lines:
                    line = line.strip()
                    if not line == "":
                        line = " ".join(line.split(" ")[:5])
                        f.write(line + "\n")

process(train_path,train_txt)
process(val_path,val_txt)

train_txt.close()
val_txt.close()


