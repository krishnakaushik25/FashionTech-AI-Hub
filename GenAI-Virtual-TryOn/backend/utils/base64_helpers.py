import base64

def array_buffer_to_base64(data: bytes) -> str:
    return base64.b64encode(data).decode('utf-8')