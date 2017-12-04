#%%
import csv
import os
os.chdir('C:\\Users\\Poul Gunnar\\PycharmProjects\\Deep-Learning')
#%%

def importer():
    x = []
    with open('output.csv', 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='\n')
        for row in reader:
            x.extend(row)
    return x

#%%
def trim(data):
    out = data.replace("              ", " ").replace("             ", " ").replace("            ", " ").replace("           ", " ").replace("          ", " ").replace("         ", " ").replace("        ", " ").replace("       ", " ").replace("      ", " ").replace("     ", " ").replace("    ", " ").replace("   ", " ").replace("  ", " ").replace(".......", ".").replace(" .. . . .", ".").replace(".....", ".").replace(".. . . .", ".").replace(" .. . . ", ".").replace(".. . . ", ".").replace(" .. . .", ".").replace(" .. . ", ".").replace(".. . ", ".").replace(" .. .", ".").replace(".. .", ".").replace("...", ".").replace(". . .", ".").replace(" .. ", ".").replace(" ..", ".").replace(".. ", ".").replace("..", ".").replace(". .", ".").replace(" . ", ". ").replace(" .",".").replace(" , ", ", ").replace(",,", ",")
    return out


def exporter(data):
    with open("output_2.0.csv", "w", encoding="utf-8") as resultFile:
        wr = csv.writer(resultFile, dialect="excel", delimiter="\n")
        wr.writerow(data)

#%%

dat = importer()
#%%
final = []
for i in range(len(dat)):
    final.append(trim(dat[i]))
#%%   
exporter(final)
#%%

