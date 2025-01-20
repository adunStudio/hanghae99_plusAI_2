import hashlib

# 파일의 해시 값 계산
def calculate_file_hash(file):
    hasher = hashlib.md5()
    hasher.update(file.read())
    file.seek(0)
    return hasher.hexdigest()