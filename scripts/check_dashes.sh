#!/usr/bin/env bash
# Hard gate on dash punctuation in article prose.
#
# The user's rule is absolute: no em-dashes, no en-dashes, no double
# hyphens used as prose punctuation. This ran as a judgment call for two
# revision passes and failed both times, so it is a check now.
#
# Usage: scripts/check_dashes.sh [files...]
#        (defaults to article/*.md plus README.md; dev-log.md is exempt,
#         it is a raw build journal and not reader-facing prose)

set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

files=("$@")
if [ ${#files[@]} -eq 0 ]; then
    mapfile -t files < <(find article -maxdepth 1 -name '*.md' ! -name 'dev-log.md')
    files+=(README.md)
fi

status=0
for f in "${files[@]}"; do
    [ -f "$f" ] || continue
    em=$(grep -c $'—' "$f" || true)
    en=$(grep -c $'–' "$f" || true)
    dh=$(grep -c -- ' -- ' "$f" || true)
    total=$((em + en + dh))
    if [ "$total" -ne 0 ]; then
        status=1
        printf 'FAIL %-24s em:%s en:%s double-hyphen:%s\n' "$f" "$em" "$en" "$dh"
        grep -n -e $'—' -e $'–' -e ' -- ' "$f" | sed 's/^/       /'
    else
        printf 'ok   %-24s\n' "$f"
    fi
done

if [ "$status" -ne 0 ]; then
    echo
    echo "Rewrite the sentence, do not swap in a comma mechanically:"
    echo "  appositive -> commas, or split into its own sentence"
    echo "  reveal     -> colon"
    echo "  aside      -> parentheses"
    echo "  dramatic pause -> delete it"
fi
exit "$status"
