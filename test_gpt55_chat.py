import requests
from azure.identity import InteractiveBrowserCredential

papyrus_endpoint = "https://westus2.papyrus.binginternal.com/chat/completions"
verify_scope = "api://5fe538a8-15d5-4a84-961e-be66cd036687/.default"

cred = InteractiveBrowserCredential()
access_token = cred.get_token(verify_scope).token
print("Token obtained successfully")

headers = {
    "Authorization": "Bearer " + access_token,
    "Content-Type": "application/json",
    "papyrus-model-name": "gpt-55-2026-04-24-Eval",
    "papyrus-quota-id": "PapyrusCustomer",
    "papyrus-timeout-ms": "100000",
}

json_body = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello in one sentence."},
    ],
    "max_completion_tokens": 50,
    "temperature": 0.3,
}

response = requests.post(
    papyrus_endpoint, headers=headers, json=json_body, verify=False
)
print(response.status_code)
print(response.text)
