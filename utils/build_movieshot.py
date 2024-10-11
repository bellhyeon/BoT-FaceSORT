from constants import END_FRAMES, SEQUENCES
import os
import shutil
import configparser

def gen_seqmaps():
    seqmap_folder = "test_data/MovieShot/seqmaps"
    os.makedirs(seqmap_folder, exist_ok=True)
    seqmap_file = "MovieShot-all.txt"
    
    with open(os.path.join(seqmap_folder, seqmap_file), "w") as f:
        f.write("name\n")
        for sequence in SEQUENCES:
            f.write(f"{sequence}\n")
    
def gen_seqinfo():
    seqinfo_root_folder = "test_data/MovieShot/MovieShot-all"
    os.makedirs(seqinfo_root_folder, exist_ok=True)
    
    for sequence in SEQUENCES:
        seqinfo = configparser.ConfigParser()

        seq_folder = os.path.join(seqinfo_root_folder, sequence)    # test_data/MovieShot/MovieShot-all/sequence
        gt_folder = os.path.join(seq_folder, "gt")
        os.makedirs(os.path.join(gt_folder), exist_ok=True) # test_data/MovieShot/MovieShot-all/sequence/gt
        org_gt_path = f"test_data/MovieShot/gt/{sequence}.txt"
        shutil.copyfile(org_gt_path, os.path.join(gt_folder, "gt.txt"))
        
        seqinfo["Sequence"] = {
            "name": sequence,
            "imDir": "img1",
            "frameRate": 24,
            "seqLength": END_FRAMES[sequence],
            "imWidth": 1280,
            "imHeight": 720,
            "imExt": ".jpg"
        }
        
        with open(os.path.join(seq_folder, "seqinfo.ini"), "w") as f:
            seqinfo.write(f)
            
if __name__ == "__main__":
    print("*** Generating seqinfo for MovieShot dataset ***")
    gen_seqinfo()
    print("*** Generating seqmaps for MovieShot dataset ***")
    gen_seqmaps()
    print("*** Done ***")