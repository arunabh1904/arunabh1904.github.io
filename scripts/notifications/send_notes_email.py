"""Send one merge summary; use Gmail Sent to deduplicate workflow retries."""
import html
import imaplib
import json
import os
import re
import smtplib
import ssl
from email.message import EmailMessage
from email.utils import formatdate
from pathlib import Path

RECIPIENT = 'arunabh1904@gmail.com'


def message_id(head):
    return f'<arxiv-notes-{head}@arunabh1904.github.io>'


def already_sent(address, password, identity):
    with imaplib.IMAP4_SSL('imap.gmail.com', 993, ssl_context=ssl.create_default_context(), timeout=30) as mailbox:
        mailbox.login(address, password)
        status, folders = mailbox.list()
        if status != 'OK':
            raise RuntimeError('Cannot inspect Sent Mail; refusing possible duplicate delivery')
        for folder in folders:
            if b'\\Sent' not in folder:
                continue
            name = re.search(rb'"([^"]+)"\s*$', folder)
            if not name:
                continue
            status, _ = mailbox.select('"' + name[1].decode() + '"', readonly=True)
            if status != 'OK':
                raise RuntimeError('Cannot open Sent Mail')
            status, matches = mailbox.uid('search', None, 'HEADER', 'Message-ID', '"' + identity + '"')
            if status != 'OK':
                raise RuntimeError('Cannot check previous delivery')
            return bool(matches[0])
        raise RuntimeError('Gmail Sent mailbox not found')


def compose(payload, sender):
    notes = payload['notes']
    added = sum(n['change'] == 'Added' for n in notes)
    updated = len(notes) - added
    counts = ', '.join(f'{count} {label}' for count, label in [(added, 'added'), (updated, 'updated')] if count)
    message = EmailMessage()
    message['From'] = sender
    message['To'] = RECIPIENT
    message['Subject'] = f'Arxiv Notes merged: {counts}'
    message['Message-ID'] = message_id(payload['head'])
    message['Date'] = formatdate(localtime=False)
    intro = 'These notes have merged to main. The website deployment may still be finishing.'
    text = [intro]
    items = []
    for note in notes:
        text.append(f"{note['change']}: {note['title']} ({note['field']})\n{note['summary']}\nNote: {note['url']}\nPaper: {note['paperUrl']}")
        n = {key: html.escape(str(value), quote=True) for key, value in note.items()}
        items.append(f"<li><p><strong>{n['change']}: {n['title']}</strong><br><span>{n['field']}</span></p><p>{n['summary']}</p><p><a href=\"{n['url']}\">Read note</a> · <a href=\"{n['paperUrl']}\">Original paper</a></p></li>")
    text.append('Merged changes: ' + payload['compareUrl'])
    message.set_content('\n\n'.join(text))
    message.add_alternative('<!doctype html><html><body><h1>Arxiv Notes: ' + html.escape(counts) + '</h1><p>' + intro + '</p><ul>' + ''.join(items) + '</ul><p><a href="' + html.escape(payload['compareUrl'], quote=True) + '">Merged changes</a></p></body></html>', subtype='html')
    return message


def deliver(payload, address, password, check=already_sent, smtp_factory=smtplib.SMTP_SSL):
    if not payload['notes']:
        return 'No note changes; no email'
    identity = message_id(payload['head'])
    if check(address, password, identity):
        return 'Already sent; skipped duplicate'
    message = compose(payload, address)
    # Never automatically retry send_message: a timeout may follow SMTP acceptance.
    with smtp_factory('smtp.gmail.com', 465, context=ssl.create_default_context(), timeout=30) as smtp:
        smtp.login(address, password)
        refused = smtp.send_message(message)
        if refused:
            raise RuntimeError('Gmail refused the notification recipient')
    return f'Sent summary for {len(payload["notes"])} notes to {RECIPIENT}'


if __name__ == '__main__':
    payload = json.loads(Path('notes-email.json').read_text())
    sender = os.environ['NOTES_GMAIL_ADDRESS']
    password = os.environ['NOTES_GMAIL_APP_PASSWORD']
    print(deliver(payload, sender, password))
