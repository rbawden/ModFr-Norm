#!/usr/bin/python
import re
import os

def read_toc(filename, unnormalised=False):
    toc = []
    headers = None
    if unnormalised:
        norm_value = 'no'
    else:
        norm_value = 'yes'
    with open(filename) as fp:
        for line in fp:
            if headers is None:
                headers = line.strip('\n').split('\t')
            else:
                info = line.strip('\n').split('\t')
                struct_info = {h: info[i] for i, h in enumerate(headers)}

                # correct dates
                struct_info['Date'] = re.sub('^\[(.+?)\]$', r'\1', struct_info['Date'])
                if '-' in struct_info['Date']:
                    struct_info['Date'] = struct_info['Date'].split('-')[0]
                    struct_info['Date'] = re.sub('^.*?(\d{4}).*?$', r'\1', struct_info['Date'])
                    #if not re.match('^\d\d\d\d$', struct_info['Date']):
                    #    print(line)
                    #    input()
                
                if struct_info['Regularisation'] == norm_value:
                    norm_file = struct_info['File'].split('/')[-1].replace('.xml', '').replace('%20', ' ')
                    struct_info['File'] = norm_file
                    toc.append(struct_info)
    return toc, headers


def get_files(txt_folder, norm_files, headers):
    for struct_info in norm_files:
        norm_file = struct_info['File']
        filename = txt_folder + '/' + norm_file + '_dAlembert.txt'
        filename2 = txt_folder + '/' + norm_file + '_dAlemBERT.txt'
        if os.path.exists(filename2):
            filename = filename2
        with open(txt_folder + '/' + norm_file + '_dAlembert.txt') as fp:
            #print(filename)
            for raw_lines in fp:
                # split on full stops followed by space and then by a capital letter (very approximative)
                lines = re.sub('\.(?= [A-ZCÉÈÀÔÖÛÜÙÎÏÂÄ])', '.\n', raw_lines.strip()).split('\n')
                for line in lines:
                    line = line.strip().replace(' ,', ',')
                    # skip markup lines
                    if re.match('<.+?>', line):
                        continue
                    if len(line.split()) > 3:
                        #print(line)
                        #print(headers)
                        #print(struct_info[headers[0]])
                        #input()
                        print(line + '\t' + '\t'.join([struct_info[h] for h in headers]))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('txt_folder')
    parser.add_argument('toc')
    parser.add_argument('-u', '--unnormalised', help='Get unnormalised instead', action='store_true')
    args = parser.parse_args()

    toc, headers = read_toc(args.toc, args.unnormalised)
    get_files(args.txt_folder, toc, headers)
