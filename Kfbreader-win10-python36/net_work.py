import wget
import tarfile

url = 'https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip'
out_fname = "datasets/PennFudanPed.zip"
file_fname = wget.download(url=url, out=out_fname)
# 提取压缩包
tar = tarfile.open(out_fname)
tar.extractall()
tar.close()
# 删除下载文件压缩包
os.remove(out_fname)