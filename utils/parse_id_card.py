import json

check_code = ["0","1","2","3","4","5","6","7","8","9","x","X"]

location_id_path = "../TianNLP/dictionary/location_id.json"

def parse_id_cards(id_card, location_id):
    if len(id_card) != 18:
        raise ValueError("身份证不符合规范")
    if id_card[17] in check_code:
        # 导入行政编码

        location = location_id[id_card[0:6]]
        birthday = id_card[6:14]
        gender = id_card[16]

        information = {
            "地区": location,
            "生日": "{}年{}月{}号".format(birthday[0:4], birthday[4:6], birthday[6:8]),
            "性别": "男" if int(gender) % 2 != 0 else "女",
        }
        return information
    else:
        raise ValueError("不合规的身份证号")

def get_location_id(path):
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


if __name__ == "__main__":
    card_id = "510311200504065230"
    information = parse_id_cards(card_id)
    print(information)



