# Security Practices

## Encryption
All customer data is encrypted at rest with AES-256 and in transit with TLS 1.3. Encryption keys are rotated every 90 days and stored in a hardware security module.

## Access Control
Production access requires single sign-on with hardware-key multi-factor authentication. Engineers receive just-in-time access that expires after 4 hours, and every session is recorded for audit.

## Incident Response
Security incidents are triaged within 1 hour. Affected customers are notified within 72 hours of a confirmed data breach, in line with GDPR requirements.
