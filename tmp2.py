import os

import yaml

noise_db = "/home/zlwang/ProtegoPlus/face_db/face_scrub/_noise_db"
protectee_db = "/home/zlwang/ProtegoPlus/face_db/face_scrub"
cluster_res = "/home/zlwang/ProtegoPlus/results/cluster/face_scrub_default_frpair0_mask0_univ_mask_ir50_adaface_casia_prot50_nokmeans.yaml"

res = {}
prot_portion = 0.5

for name in os.listdir(noise_db):
    if name.startswith(('.', '_')):
        continue
    personal_path = os.path.join(noise_db, name)
    imgs = [os.path.join(personal_path, x) for x in os.listdir(personal_path) if x.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')) and not x.startswith(('.', '_'))]
    res[name] = imgs

for name in os.listdir(protectee_db):
    if name.startswith(('.', '_')):
        continue
    personal_path = os.path.join(protectee_db, name)
    imgs = [os.path.join(personal_path, x) for x in os.listdir(personal_path) if x.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')) and not x.startswith(('.', '_'))]
    orig_num = int(len(imgs) * (1 - prot_portion))
    orig_imgs = imgs[:orig_num]
    prot_imgs = imgs[orig_num:]
    prot_imgs = [x + "/prot" for x in prot_imgs]
    res[name] = orig_imgs + prot_imgs

with open(cluster_res, 'w') as f:
    yaml.dump(res, f)