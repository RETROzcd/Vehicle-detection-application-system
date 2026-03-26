import re


final_valid_plates = []

def processplatefunc():
    from plate_pipeline import legacy_dp as dp

    dp.dpfunc()
    resultplate_array = dp.resultplate_array
    #print(resultplate_array)
    # 验证车牌的正则表达式
    def validate_chinese_license_plate(plate):
        # 普通车牌正则表达式
        regular_plate_pattern = r'^[\u4e00-\u9fa5][A-Z][A-Z0-9]{5}$'
        # 新能源车牌正则表达式
        new_energy_plate_pattern = r'^[\u4e00-\u9fa5][A-Z][A-Z0-9]{6}$'
        if re.match(regular_plate_pattern, plate):
            return "普通车牌"
        elif re.match(new_energy_plate_pattern, plate):
            return "新能源车牌"
        else:
            return "无效车牌"
    # 提取车牌号部分的函数
    def extract_license_plate(plate_str):
        match = re.match(r'^[^\s]+', plate_str)
        if match:
            return match.group(0)
        return None

    # 提取并验证车牌号
    #final_valid_plates = []
    extracted_plates = [extract_license_plate(plate) for plate in resultplate_array]
    #print(extracted_plates)

    #for plate in extracted_plates:
    #    print(f"{plate}: {validate_chinese_license_plate(plate)}")

    for plate in extracted_plates:
        if validate_chinese_license_plate(plate) != "无效车牌":
            final_valid_plates.append(plate)

    return final_valid_plates
    #print(final_valid_plates)#最终

'''
processplatefunc()
print(final_valid_plates)
'''

