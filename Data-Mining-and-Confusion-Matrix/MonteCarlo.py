import random

def load_data():
    lines = None

    #dosya yukle
    with open("./Triyaj.csv","r",encoding="utf-8") as f:
        lines = f.readlines()
    return lines

def convert_lines_to_rows(lines):

    #header ve satirlar olarak böl
    data = []
    for line in lines:
        columns = line.strip().replace('"','').split(";")
        columns = [col.strip() for col in columns]
        data.append( columns )

    header = data[0]
    rows = data[1:]

    return rows


def convert_rows_to_diagnostic_symptom_map(rows):
    #hastalik adları ve belirtileri tekilleştir
    symptoms = set()
    diagnostic_symptom_map = dict()
    for row in rows:
        symptoms.add( row[1] )

        if row[0] not in diagnostic_symptom_map:
            diagnostic_symptom_map[ row[0] ] = dict()

        diagnostic_symptom_map[row[0]][row[1]] = float(row[2])




    #hastalik satir, belirtiler sutun
    symptom_list = sorted( list( symptoms ) )

    return ( symptom_list, diagnostic_symptom_map )


def prepare_sym(symptom_list, diagnostic_symptom_map, count):

    diagnostics = sorted( list( diagnostic_symptom_map.keys() ) )
    header = []
    header.extend( symptom_list)
    header.append("TANI")
    header.append("TANI_IDX")


    rows = []
    rows.append(header)
    for i in range(0,count):
        row = [0] * len(header)
        sel_diagnostic = diagnostics[ random.randint(0,len( diagnostics ))-1 ]

        #taniyi ata
        row[-2] = sel_diagnostic
        row[-1] = diagnostics.index(sel_diagnostic)

        for prop in diagnostic_symptom_map[sel_diagnostic]:
            prop_index = header.index(prop)
            prop_val = diagnostic_symptom_map[sel_diagnostic][prop]
            sym_val = 1 if random.random() <= prop_val else 0
            row[prop_index] = sym_val
        rows.append(row )




    return  rows


def main(count):
    lines = load_data()
    rows = convert_lines_to_rows(lines)
    symptom_list, diagnostic_symptom_map = convert_rows_to_diagnostic_symptom_map(rows)
    symdata = prepare_sym(symptom_list, diagnostic_symptom_map, count)

    with open("./sym_data.csv", mode="w", encoding="utf-8") as file:
        for sym_row in symdata:
            sym_row = [str(i) for i in sym_row]
            file.write( ";".join(sym_row) + "\n" )


if __name__ == '__main__':
    main(count = 1000)