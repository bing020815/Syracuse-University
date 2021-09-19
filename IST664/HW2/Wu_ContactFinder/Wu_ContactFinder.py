"""
This program was adapted from the Stanford NLP class SpamLord homework assignment.
    The code has been rewritten and the data modified, nevertheless
    please do not make this code or the data public.
This base version has no patterns, but has two patterns suggested in comments
    in order to get you started .
"""
import sys
import os
import re
import pprint

"""
TODO
For Part 1 of our assignment, add to these two lists of patterns to match
examples of obscured email addresses and phone numbers in the text.
For optional Part 3, you may need to add other lists of patterns.
"""
# email .edu patterns

# each regular expression pattern should have exactly two sets of parentheses.
#   the first parenthesis should be around the someone part
#   the second parenthesis should be around the somewhere part
#   in an email address whose standard form is someone@somewhere.edu
epatterns = []
epatterns.append(r'([A-Za-z.]+)@([A-Za-z.]+)\.edu') 
    # 1. Match standard emails with a ‘period’ sign in the 'name' group or 'domain' group --19
epatterns.append(r'([A-Za-z.]+)\s{1,3}@\s{1,3}([A-Za-z.]+)\.edu') 
    # 2. Match emails with ' @ ' instead of '@' --23(4) 
epatterns.append(r'([A-Za-z.]+)@([A-Za-z.]+)\.[EDU]+') 
    # 3. Match emails with capital 'EDU' -- 24(1) 
epatterns.append(r'([a-zA-Z-]+)-@-([a-z-A-Z]+)-\.[eduEDU-]+') 
    # 4. Match emails address with dash sign within characters --24 +[1]
epatterns.append(r'[\s>]([a-zA-Z\s]+)\sat\s([a-zA-Z\s]+)\sedu')
    # 5. Match emails with 'dot' instead of '.'; 'at' instead of '@' -- 24 +1 +[4]
epatterns.append(r'([A-Za-z]+)\s[atAT]+\s([A-Za-z;]+);[edu]+') 
    # 6. Match emails with ';' sign instead  of '.'; 'at' instead of '@' -- 24 +1 +4 +[1]
epatterns.append(r"([a-zA-Z]+)\s\+\s'@'\s\+\s([a-zA-Z]+)") 
    # 7. Match emails with the format of "name + '@' + domain" -- 24 +1 +4 +1 +[1]
epatterns.append(r"\s([a-z]+)\sat\s([a-zA-Z.]+)\.edu") 
    # 8. Match emails with ' at ' instead of '@' -- 25(1) +1 +4 +1 +1
epatterns.append(r"([a-zA-Z]+)<del>@([a-zA-Z.]+)\.edu") 
    # 9. Match emails with '<del>@' instead of '@' -- 28(3) +1 +4 +1 +1
epatterns.append(r"([a-zA-Z]+)&#x40;([a-zA-Z.]+)\.edu") 
    # 10. Match emails with '&#x40;' instead of '@' -- 30(2) +1 +4 +1 +1
epatterns.append(r"([a-zA-Z]+)\s<at\ssymbol>\s([a-zA-Z.]+)\.edu") 
    # 11. Match emails with ' <at symbol> ' instead of '@' -- 32(2) +1 +4 +1 +1
epatterns.append(r"([a-zA-Z.]+)\s\(followed\sby\s&ldquo;@([a-zA-Z.]+)\.edu") 
    # 12. Match emails with ' (followed by &ldquo;@' instead of '@' -- 33(1) +1 +4 +1 +1
epatterns.append(r"([a-zA-Z.]+)\s\(followed\sby\s\"@([a-zA-Z.]+)\.edu") 
    # 13. Match emails with ' (followed by "@' instead of '@' -- 34(1) +1 +4 +1 +1
epatterns.append(r"[\s>]([a-zA-Z\s]+)\sAT\s([a-zA-Z\s]+)\sDOT\sedu") 
    # 14. Match emails with 'DOT' instead of '.'; 'AT' instead of '@' -- 35(1) +1 +4 +1 +1
epatterns.append(r"([a-zA-Z]+)\sat\s([a-zA-Z]+)\sdt\scom") 
    # 15. Match emails with 'gradiance dt com' instead of 'stanford.edu' -- 35 +1 +4 +1 +1 +[1]
epatterns.append(r"([a-zA-Z]+)\sat\s[<die!\s\->]+([a-z]+)\s[a-z<>!\-\s]+\sedu") 
    # 16. Match emails with 'tags', 'at' and 'dot'-- 36(1) +1 +4 +1 +1 +1 
epatterns.append(r"([a-zA-Z]+)\sWHERE\s([a-zA-Z]+)\s\DOM\sedu") 
    # 17. Match emails with ' WHERE ' and ' DOM ' instead of '@' and '.' -- 37(1) +1 +4 +1 +1 +1 

# phone patterns
# each regular expression pattern should have exactly three sets of parentheses.
#   the first parenthesis should be around the area code part XXX
#   the second parenthesis should be around the exchange part YYY
#   the third parenthesis should be around the number part ZZZZ
#   in a phone number whose standard form is XXX-YYY-ZZZZ
ppatterns = []
ppatterns.append(r'(\d{3})-(\d{3})-(\d{4})')
    # 18. Match phone with format ___-___-____ -- 45 +19
ppatterns.append(r'\((\d{3})\)(\d{3})-(\d{4})')
    # 19. Match phone with format (___)___-____ -- 45 +27
ppatterns.append(r'\((\d{3})\)\s(\d{3})-(\d{4})')
    # 20. Match phone with format (___) ___-____ -- 45 +66
ppatterns.append(r'\+1\s(\d{3})\s(\d{3})\s(\d{4})')
    # 21. Match phone with format +1 ___ ___ ____ -- 45 +68
ppatterns.append(r'\[(\d{3})\]\s(\d{3})-(\d{4})')
    # 22. Match phone with format [___] ___-____ -- 45 +70
ppatterns.append(r'\+1\s(\d{3})\s(\d{3})-(\d{4})')
    # 23. Match phone with format +1 ___ ___-____ -- 45 +68
""" 
This function takes in a filename along with the file object and
scans its contents against regex patterns. It returns a list of
(filename, type, value) tuples where type is either an 'e' or a 'p'
for e-mail or phone, and value is the formatted phone number or e-mail.
The canonical formats are:
     (name, 'p', '###-###-#####')
     (name, 'e', 'someone@something')
If the numbers you submit are formatted differently they will not
match the gold answers

TODO
For Part 3, if you have added other lists, you should add
additional for loops that match the patterns in those lists
and produce correctly formatted results to append to the res list.
"""
def process_file(name, f):
    # note that debug info should be printed to stderr
    # sys.stderr.write('[process_file]\tprocessing file: %s\n' % (path))
    res = []
    for line in f:
        # you may modify the line, using something like substitution
        #    before applyting the patterns

        # email pattern list
        for epat in epatterns:
            # each epat has 2 sets of parentheses so each match will have 2 items in a list
            matches = re.findall(epat,line)
            for m in matches:
                # string formatting operator % takes elements of list m
                #   and inserts them in place of each %s in the result string
                # email has form  someone@somewhere.edu
                #email = '%s@%s.edu' % m
                
                if '-' in m[0] or '-' in m[1]: ## total +1
                    ## Substitute the '-' sign with None
                    ## ex: d-l-w-h-@-s-t-a-n-f-o-r-d-.-e-d-u
                    ## matched with pattern 4: r'([a-zA-Z-]+)-@-([a-z-A-Z]+)-\.[eduEDU-]+'
                    email = '{}@{}.edu'.format(re.sub(r'-','',m[0]),re.sub(r'-','',m[1]))
                    res.append((name,'e',email))

                elif 'dot' in m[1]: ## total +3
                    ## Replace ' dot ' with '.'
                    ## hager at cs dot jhu dot edu / serafim at cs dot stanford dot edu
                    ##  uma at cs dot stanford dot edu
                    ## matched with pattern 5: r'[\s>]([a-zA-Z\s]+)\sat\s([a-zA-Z\s]+)\sedu'
                    domain_temp = re.sub(r' dot ','.',m[1])
                    domain= re.sub(r' dot','',domain_temp)
                    ename = m[0].strip()
                    email = '{}@{}.edu'.format(ename,domain)
                    res.append((name,'e',email))
                    if m[0] == 'subh':
                        ## uma at cs dot stanford dot edu / subh AT stanford DOT edu +1
                        email = '{}@{}.edu'.format(m[0],'cs.'+ m[1])
                        res.append((name,'e',email))
                        email2 = '{}@{}.edu'.format(name,m[1])
                        res.append((name,'e',email2))
                    
                elif 'cs ' in m[1]: ## total +1
                    ## Replace ' ' with '.' 
                    ## ex: pal at cs stanford edu +1
                    ## matched with pattern 5: r'[\s>]([a-zA-Z\s]+)\sat\s([a-zA-Z\s]+)\sedu' 
                    domain = re.sub(r' ','.',m[1])
                    email = '{}@{}.edu'.format(m[0], domain)
                    res.append((name,'e',email))


                elif ';' in m[1]: ## total +1
                    ## Substitute the ';' sign with '.'
                    ## ex: jks at robotics;stanford;edu
                    ## matched with pattern 6: r'([A-Za-z]+)\s[atAT]+\s([A-Za-z;]+);[edu]+'
                    email = '{}@{}.edu'.format(re.sub(r'-','',m[0]),re.sub(r';','.',m[1]))
                    res.append((name,'e',email))

                elif 'domain' in m[1]: ## total +1
                    ## email assigned with variables instead of traditional obscured email
                    ## ex: name + '@' + domain 
                    ## matched with pattern 7: r"([a-zA-Z]+)\s\+\s'@'\s\+\s([a-zA-Z]+)"
                    email = '{}@{}.edu'.format(name,'stanford')
                    res.append((name,'e',email))
    
                elif 'gradiance' in m[1]: ## total +1
                    ## save email as '.com' instead of 'edu'
                    ## ex: support at gradiance dt com
                    ## matched with pattern 15: r"([a-zA-Z]+)\sat\s([a-zA-Z]+)\sdt\scom"
                    email = '{}@{}.com'.format(m[0],m[1])
                    res.append((name,'e',email))

                else:
                    email = '{}@{}.edu'.format(m[0],m[1])
                    res.append((name,'e',email))


        # phone pattern list
        for ppat in ppatterns:
            # each ppat has 3 sets of parentheses so each match will have 3 items in a list
            matches = re.findall(ppat,line)
            for m in matches:
                # phone number has form  areacode-exchange-number
                #phone = '%s-%s-%s' % m
                phone = '{}-{}-{}'.format(m[0],m[1],m[2])
                res.append((name,'p',phone))
    return res

"""
You should not edit this function.
"""
def process_dir(data_path):
    # save complete list of candidates
    guess_list = []
    # save list of filenames
    fname_list = []

    for fname in os.listdir(data_path):
        if fname[0] == '.':
            continue
        fname_list.append(fname)
        path = os.path.join(data_path,fname)
        f = open(path,'r', encoding='latin-1')
        # get all the candidates for this file
        f_guesses = process_file(fname, f)
        guess_list.extend(f_guesses)
    return guess_list, fname_list

"""
You should not edit this function.
Given a path to a tsv file of gold e-mails and phone numbers
this function returns a list of tuples of the canonical form:
(filename, type, value)
"""
def get_gold(gold_path):
    # get gold answers
    gold_list = []
    f_gold = open(gold_path,'r', encoding='latin-1')
    for line in f_gold:
        gold_list.append(tuple(line.strip().split('\t')))
    return gold_list

"""
You should not edit this function.
Given a list of guessed contacts and gold contacts, this function
    computes the intersection and set differences, to compute the true
    positives, false positives and false negatives. 
It also takes a dictionary that gives the guesses for each filename, 
    which can be used for information about false positives. 
Importantly, it converts all of the values to lower case before comparing.
"""
def score(guess_list, gold_list, fname_list):
    guess_list = [(fname, _type, value.lower()) for (fname, _type, value) in guess_list]
    gold_list = [(fname, _type, value.lower()) for (fname, _type, value) in gold_list]
    guess_set = set(guess_list)
    gold_set = set(gold_list)

    # for each file name, put the golds from that file in a dict
    gold_dict = {}
    for fname in fname_list:
        gold_dict[fname] = [gold for gold in gold_list if fname == gold[0]]

    tp = guess_set.intersection(gold_set)
    fp = guess_set - gold_set
    fn = gold_set - guess_set

    pp = pprint.PrettyPrinter()
    #print 'Guesses (%d): ' % len(guess_set)
    #pp.pprint(guess_set)
    #print 'Gold (%d): ' % len(gold_set)
    #pp.pprint(gold_set)

    print ('True Positives (%d): ' % len(tp))
    # print all true positives
    pp.pprint(tp)
    print ('False Positives (%d): ' % len(fp))
    # for each false positive, print it and the list of gold for debugging
    for item in fp:
        fp_name = item[0]
        pp.pprint(item)
        fp_list = gold_dict[fp_name]
        for gold in fp_list:
            s = pprint.pformat(gold)
            print('   gold: ', s)
    print ('False Negatives (%d): ' % len(fn))
    # print all false negatives
    pp.pprint(fn)
    print ('Summary: tp=%d, fp=%d, fn=%d' % (len(tp),len(fp),len(fn)))

"""
You should not edit this function.
It takes in the string path to the data directory and the gold file
"""
def main(data_path, gold_path):
    guess_list, fname_list = process_dir(data_path)
    gold_list =  get_gold(gold_path)
    score(guess_list, gold_list, fname_list)

    count=0
    for n, t, p in gold_list:
        if t == 'e':
            count +=1  
    print('Total eamil count:{}'.format(count))

    pcount=0
    for n, t, p in gold_list:
        if t == 'p':
            pcount +=1  
    print('Total phone count:{}'.format(pcount))

"""
commandline interface assumes that you are in the directory containing "data" folder
It then processes each file within that data folder and extracts any
matching e-mails or phone numbers and compares them to the gold file
"""
if __name__ == '__main__':
    print ('Assuming ContactFinder.py called in directory with data folder')
    main('data/dev', 'data/devGOLD')
