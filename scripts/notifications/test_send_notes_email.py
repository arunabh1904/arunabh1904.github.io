import unittest
from unittest.mock import MagicMock
from send_notes_email import compose, deliver, message_id

PAYLOAD = {'head': 'a' * 40, 'compareUrl': 'https://github.com/example/compare/a...b', 'notes': [
    {'title': 'Maps <script> & topology', 'field': 'Mapping', 'change': 'Added', 'summary': 'Evidence & limits.',
     'url': 'https://example.com/note', 'paperUrl': 'https://arxiv.org/abs/2505.12246'}]}


class DeliveryTests(unittest.TestCase):
    def test_summary_contains_both_links_and_correct_recipient(self):
        message = compose(PAYLOAD, 'sender@gmail.com')
        self.assertEqual(message['To'], 'arunabh1904@gmail.com')
        self.assertEqual(message['Message-ID'], message_id(PAYLOAD['head']))
        self.assertIn('1 added', message['Subject'])
        body = message.get_body(preferencelist=('html',)).get_content()
        self.assertIn('https://arxiv.org/abs/2505.12246', body)
        self.assertIn('https://example.com/note', body)
        self.assertIn('&lt;script&gt;', body)
        self.assertNotIn('<script>', body)

    def test_empty_and_previously_sent_batches_do_not_connect(self):
        smtp = MagicMock()
        self.assertIn('no email', deliver({'notes': []}, 'sender', 'secret', smtp_factory=smtp))
        self.assertIn('duplicate', deliver(PAYLOAD, 'sender', 'secret', check=lambda *args: True, smtp_factory=smtp))
        smtp.assert_not_called()

    def test_success_sends_one_email(self):
        smtp = MagicMock()
        connection = smtp.return_value.__enter__.return_value
        connection.send_message.return_value = {}
        self.assertIn('Sent summary', deliver(PAYLOAD, 'sender', 'secret', check=lambda *args: False, smtp_factory=smtp))
        connection.send_message.assert_called_once()

    def test_ambiguous_delivery_is_not_retried(self):
        smtp = MagicMock()
        connection = smtp.return_value.__enter__.return_value
        connection.send_message.side_effect = TimeoutError('delivery uncertain')
        with self.assertRaises(TimeoutError):
            deliver(PAYLOAD, 'sender', 'secret', check=lambda *args: False, smtp_factory=smtp)
        connection.send_message.assert_called_once()

    def test_deduplication_failure_blocks_send(self):
        smtp = MagicMock()
        check = MagicMock(side_effect=RuntimeError('Sent unavailable'))
        with self.assertRaises(RuntimeError):
            deliver(PAYLOAD, 'sender', 'secret', check=check, smtp_factory=smtp)
        smtp.assert_not_called()


if __name__ == '__main__':
    unittest.main()
