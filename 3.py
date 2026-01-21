import smbclient

# 连接到Windows共享文件夹
with smbclient.open_file(
    r"\\WindowsPC\video_clip\目标文件.mp4", 
    mode="wb", 
    username="Windows用户名", 
    password="Windows密码"
) as f:
    # 从Linux读取文件并上传
    with open("/mnt/wd0/la/源文件.mp4", "rb") as src:
        f.write(src.read())