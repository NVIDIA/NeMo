new_file = open("new_kb.tsv", "w")
old_file = open("tiny_example_test_kb.csv", "r", encoding='utf-8-sig')

for line in old_file.readlines():
    cid, concept = line.split(",")
    cid = int(cid)
    new_file.write(f"{cid}\t{concept}")
    print(cid, concept)

new_file.close()
