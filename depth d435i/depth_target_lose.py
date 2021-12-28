#임시변수
following_pers = 0
distance_val = 0
id = 0


# 기본선언
# 타겟의 폐색 거리저장
save_targetdis = 0
# 타겟종료
searchCountFlag = 0

# 영상입력 한프레임 for문시작할떄
while True:
    searchId_dic = {}

    # 타겟추적 for문 시작할떄 Process detections 딕셔너리에 id=거리 저장
    searchId_dic[id] = distance_val

    # 리스트저장 추적성공시 distance_val 값 임시 저장
    if following_pers == id:
        # 타겟의 폐색대비 거리저장
        save_trgtDst = distance_val
        # 타겟 추적성공시 플래그초기화
        searchCountFlag = 0

    # following_pers가 추적이 안되고 있으면 (딕셔너리 키안에 id 추가가 안되있으면)
    if not following_pers in searchId_dic:
        searchCountFlag += 1
        # 15프레임이하, 5프레임이상 없었으면
        if 15 > searchCountFlag > 5:
            # 리스트내에 있는 비슷한 거리의 타겟을 찾아 변경
            for searchId, searchDis in searchId_dic.items():
                if 10.0 + save_trgtDst > searchDis > save_trgtDst - 10.0 :
                    # 타겟변경
                    following_pers = searchId





