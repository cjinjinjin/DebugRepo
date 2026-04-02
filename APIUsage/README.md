# Papyrus API Setup Guide

Guide for accessing Papyrus (gpt-image-1 and other models) on a new machine.

---

## Prerequisites

- Microsoft corporate account (e.g. `jinjinchen@microsoft.com`)
- Azure CLI installed: https://aka.ms/install-azure-cli
- Python 3.8+

---

## Step 1: Request Papyrus Access

Join the **Papyrus Customer Role** security group via MyAccess:

1. Go to https://idweb.microsoft.com/IdentityManagement/aspx/groups/AllGroups.aspx
2. Search for **"Papyrus Customer Role"**
3. Request access and wait for approval (can take several hours)

This grants freemium quota to all Papyrus models using your AAD token.

---

## Step 2: Install Python Dependencies

```bash
pip install azure-identity azure-identity-broker requests
```

---

## Step 3: Azure Login

Log in using the **Microsoft internal tenant**:

```bash
az login --tenant "72f988bf-86f1-41af-91ab-2d7cd011db47"
```

### How to find your Tenant ID

The tenant ID `72f988bf-86f1-41af-91ab-2d7cd011db47` is Microsoft's corporate Azure AD tenant. You can verify or retrieve it in several ways:

**Option A — Azure CLI (after login):**
```bash
az account show --query tenantId -o tsv
```

**Option B — Azure Portal:**
1. Go to https://portal.azure.com
2. Search for **"Azure Active Directory"**
3. The **Tenant ID** is shown on the Overview page

**Option C — az login output:**
When you run `az login` without `--tenant`, the output lists all tenants. Look for the one named **"Microsoft"** — that tenant's ID is `72f988bf-86f1-41af-91ab-2d7cd011db47`.

---

## Step 4: Run the Image Generation Script

```bash
python papyrus_generate_image.py --prompt "a cat sitting on a cloud" --output output.png
```

Optional arguments:

| Argument | Default | Options |
|---|---|---|
| `--size` | `1024x1024` | `1024x1024`, `1792x1024`, `1024x1792` |
| `--quality` | `standard` | `standard`, `hd` |
| `--n` | `1` | number of images |

---

## How Authentication Works

The script uses `AzureCliCredential` from the `azure-identity` package. It reads the token cached by `az login` and requests a scoped token for Papyrus:

```python
from azure.identity import AzureCliCredential

credential = AzureCliCredential()
token = credential.get_token("api://5fe538a8-15d5-4a84-961e-be66cd036687/.default")
```

The scope `api://5fe538a8-15d5-4a84-961e-be66cd036687/.default` is the Papyrus internal application ID.

---

## API Endpoints

| Purpose | Endpoint |
|---|---|
| Image generation | `https://westus2large.papyrus.binginternal.com/images/generations` |
| Image editing | `https://westus2large.papyrus.binginternal.com/images/edits` |
| Chat completions | `https://westus2.papyrus.binginternal.com/chat/completions` |

---

## Troubleshooting

**Token error / AADSTS65002:**
- Make sure you logged in with the correct tenant: `az login --tenant 72f988bf-86f1-41af-91ab-2d7cd011db47`
- Run `az logout` first if switching accounts

**401 / 403 Unauthorized:**
- Your account may not have Papyrus access yet — wait for MyAccess approval
- Try `az logout && az login --tenant 72f988bf-86f1-41af-91ab-2d7cd011db47`

**Token expired:**
- Re-run `az login --tenant 72f988bf-86f1-41af-91ab-2d7cd011db47`

**`azure-identity-broker` warning:**
- Install with: `pip install azure-identity-broker`
- This is optional but suppresses broker-related warnings on Windows
