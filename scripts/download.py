from figshare.figshare import Figshare
import os

data =  {
        "Events"        :   7770599,
        "Matches"       :   7770422,
        "Competitions"  :   7765316,
        "Teams"         :   7765310,
        "Players"       :   7765196,
        "Referees"      :   8082665,
        "Coaches"       :   8082650,
        "Event names"   :   1174383,
        "Tag names"     :   11743818
        }

def download():
    if os.name != 'posix':
        raise Exception("Error: this script (download.py) is intended for use on POSIX-compliant systems.")

    os.system("rm -r ../data")
    os.system("mkdir ../data")

    print("Emptied data folder")

    fs = Figshare()

    for article_name in data.keys():

        article_id = data[article_name]

        number_of_files = len(fs.list_files(article_id))

        fs.retrieve_files_from_article(article_id,"../data")

        print(f'Downloaded {number_of_files} files from article "{article_name}" (#{article_id})')

    os.system("mkdir ../data/figshare")
    os.system("mv ../data/figshare_*/* ../data/figshare/")
    os.system("rm -r ../data/figshare_*")

    print("Collated files")

    os.system('unzip "../data/figshare/*.zip" -d "../data/figshare/"')

    print("Unzipped files")

if __name__ == "__main__":
    download()
