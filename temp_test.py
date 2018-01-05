import re
string = 'asdf/b3232331/b3a1sd3213sad1f3d'
pattern=re.compile(pattern="(?![/b])")

pattern2=re.compile(pattern=r"b/n")
result=re.sub(pattern=pattern,repl="/n",string=string)
print(result)

result2=re.sub(pattern=pattern2,repl="b",string=result)
print(result2)
