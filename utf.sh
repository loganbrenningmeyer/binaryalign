cd datasets/en-fr/English-French/trial

for f in *.e; do
  if file -I "$f" | grep -qi "iso-8859"; then
    iconv -f ISO-8859-1 -t UTF-8 "$f" > "$f.tmp" && mv "$f.tmp" "$f"
  fi
done