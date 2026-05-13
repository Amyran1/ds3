#!/usr/bin/env bash
set -euo pipefail

INPUT=$(cat)
TOOL_NAME=$(jq -r '.tool_name // ""' <<< "$INPUT")
FILE_PATH=$(jq -r '.tool_input.file_path // ""' <<< "$INPUT")
COMMAND=$(jq -r '.tool_input.command // ""' <<< "$INPUT")

DENIED_LIST="$(dirname "$0")/../prompts/denied_writes.txt"

deny() {
    local reason="$1"
    jq -n --arg r "$reason" '{
        hookSpecificOutput: {
            hookEventName: "PreToolUse",
            permissionDecision: "deny",
            permissionDecisionReason: $r
        }
    }'
    exit 2
}

case "$TOOL_NAME" in
    Edit|Write|MultiEdit|NotebookEdit)
        if [[ -n "$FILE_PATH" && -s "$DENIED_LIST" ]]; then
            while IFS= read -r pattern; do
                [[ -z "$pattern" || "$pattern" =~ ^# ]] && continue
                if [[ "$FILE_PATH" =~ $pattern ]]; then
                    deny "autoloop: writes to '$FILE_PATH' are not allowed in this session (matched: $pattern)"
                fi
            done < "$DENIED_LIST"
        fi
        ;;
    Bash)
        if [[ "$COMMAND" =~ ^[[:space:]]*git[[:space:]]+(commit|push|reset|checkout|clean|stash)\b ]]; then
            deny "autoloop: git mutation commands are reserved for the supervisor"
        fi
        if [[ -s "$DENIED_LIST" ]]; then
            while IFS= read -r pattern; do
                [[ -z "$pattern" || "$pattern" =~ ^# ]] && continue
                if [[ "$COMMAND" =~ \>+[[:space:]]*([^[:space:]\|]+) ]]; then
                    target="${BASH_REMATCH[1]}"
                    if [[ "$target" =~ $pattern ]]; then
                        deny "autoloop: shell redirect to '$target' is not allowed (matched: $pattern)"
                    fi
                fi
            done < "$DENIED_LIST"
        fi
        ;;
esac

exit 0
