#!/usr/bin/env bash

OUT="all_polymarket_events.json"
TMP="tmp_events.jsonl"
> "$TMP"

LIMIT=1000
OFFSET=0
PAGE=0
TOTAL_EVENTS=0
BAR_WIDTH=40

echo "Fetching ALL Polymarket events..."

while :; do
  DATA=$(curl -s "https://gamma-api.polymarket.com/events?limit=$LIMIT&offset=$OFFSET")

  if [ -z "$DATA" ]; then
    echo -e "\nEmpty API response. Stopping."
    break
  fi

  COUNT=$(echo "$DATA" | jq 'length')

  if [ "$COUNT" -eq 0 ]; then
    break
  fi

  echo "$DATA" >> "$TMP"

  PAGE=$((PAGE+1))
  OFFSET=$((OFFSET + LIMIT))
  TOTAL_EVENTS=$((TOTAL_EVENTS + COUNT))

  FILLED=$((PAGE % BAR_WIDTH))
  EMPTY=$((BAR_WIDTH - FILLED))
  BAR=$(printf "%${FILLED}s" | tr ' ' '#')
  SPACE=$(printf "%${EMPTY}s")

  printf "\r[%s%s] Pages: %d | Events: %d | Offset: %d" "$BAR" "$SPACE" "$PAGE" "$TOTAL_EVENTS" "$OFFSET"
done

echo -e "\nMerging pages into final JSON..."
jq -s 'add' "$TMP" > "$OUT"
rm "$TMP"

echo "Done. Saved to $OUT"
