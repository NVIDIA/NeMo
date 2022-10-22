import sys


def convert_dummy(t):
    if t == "DUMMY1":
        return "<DELETE>"
    elif t == "DUMMY2":
        return "<DELETE> <DELETE>"
    elif t == "DUMMY3":
        return "<DELETE> <DELETE> <DELETE>"
    elif t == "DUMMY4":
        return "<DELETE> <DELETE> <DELETE> <DELETE>"
    else:
        assert False, "unknown tag: " + t

inpname, outname = sys.argv[1], sys.argv[2]

out = open(outname, "w", encoding="utf-8")

with open(inpname, "r", encoding="utf-8") as f:
    for line in f:
        prediction_str, input_str, _, _, semiotic_prediction_str = line.split("\t")
        predictions = prediction_str.split(" ")
        inputs = input_str.split(" ")
        if len(predictions) != len(inputs):
            print ("WARNING: number of tokens mismatch: len(predictions)=" + str(len(predictions)) + "; len(inputs)=" + str(len(inputs)) + "; line=" + line)
        output_str = ""
        assert(inputs[0] == "<bos>")
        if predictions[0].startswith("DUMMY"):
            output_str += convert_dummy(predictions[0]) + " "            
        for t, w in zip(predictions[1:], inputs[1:]):
            if t == "SPACE":
                output_str += w + " "
            elif t == "JOIN":
                output_str += w
            else:
                if t.startswith("DUMMY"):
                    output_str += w + " " + convert_dummy(t) + " "
                else:
                    print ("unknown tag: " + t + "; will truncate line=" + line)
                    output_str += w + " "  # regard as SPACE
        out.write(output_str.strip() + "\n")
out.close()
