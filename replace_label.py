from pathlib import Path
import re


""" Replace the labels in a file with existing labeled data in SVM format. """

def main(fin, freplace, fout):
    pin = Path(fin).resolve()
    preplace = Path(freplace).resolve()
    pout = Path(fout).resolve()

    with open(pin, encoding='utf-8') as fhin, \
         open(preplace, encoding='utf-8') as fhreplace, \
         open(pout, 'w', encoding='utf-8') as fhout:
        for orig, repl in zip(fhin, fhreplace):
            repl = repl.strip()
            replaced = re.sub(r'^([^\s]+)', repl, orig)
            fhout.write(replaced)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Replace the labels in file with SVM-feats format.')
    parser.add_argument('fin', help='path to original file.')
    parser.add_argument('freplace', help='path to replacement labels.')
    parser.add_argument('fout', help='path to output file.')

    args = parser.parse_args()

    main(args.fin, args.freplace, args.fout)
