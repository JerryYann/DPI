from deepface import DeepFace

score = 0
acc = 0

for idx in range(0, 1000):
    fname = str(idx).zfill(5)
    fname1 = str(idx-1).zfill(5)

    # img1_path = f'F10004x-tmp2/{fname}.png'
    img1_path = f'/data/yangjiarui/project/ILVR/F1000ilvr16x/{fname}.png'
    img2_path = f'/data/yangjiarui/project/datasets/FFHQ1000/{fname}.png'
    result = DeepFace.verify(img1_path=img1_path, img2_path=img2_path, enforce_detection=False)
    acc += result['verified']
    score += result['distance']
    print(result['distance'])

print('avg_score: ',  score /1000)
print('acc: ',  acc /1000)
