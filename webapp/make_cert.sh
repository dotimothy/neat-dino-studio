#!/usr/bin/env bash
# Make the self-signed certificate the web demo serves HTTPS with (browsers allow the
# webcam only on HTTPS pages). Used by webapp/run.sh on start and by setup.sh.
#   webapp/make_cert.sh [CERT_DIR]     # default: webapp/certs next to this script
# Does nothing when CERT_DIR already holds cert.pem and key.pem. Exits 1 when openssl
# is missing or fails, 0 otherwise.
set -euo pipefail
CERTS="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/certs}"
[[ -f "$CERTS/cert.pem" && -f "$CERTS/key.pem" ]] && exit 0
command -v openssl > /dev/null || { echo "openssl not found; HTTPS (and so the webcam) stays off" >&2; exit 1; }
mkdir -p "$CERTS"
san="DNS:localhost,DNS:$(hostname),IP:127.0.0.1"
for ip in $(hostname -I 2> /dev/null); do san="$san,IP:$ip"; done
if openssl req -x509 -newkey rsa:2048 -nodes -days 825 -subj "/CN=Neat DINO Studio" \
     -addext "subjectAltName=$san" -keyout "$CERTS/key.pem" -out "$CERTS/cert.pem" 2> /dev/null; then
  chmod 600 "$CERTS/key.pem"
  echo "made a self-signed certificate for $san"
else
  rm -f "$CERTS/key.pem" "$CERTS/cert.pem"
  echo "openssl could not make a certificate in $CERTS" >&2
  exit 1
fi
