# Julio Cesar Reis Filho — Portfolio MCP Server ⚡

Official Model Context Protocol (MCP) server for **Julio Cesar Reis Filho** — Senior Systems Architect & LLM Engineer.

Allows AI agents, Claude Desktop, Cursor, and automated evaluators to query verified technical dossiers, inspect the **503M CIR-Engine Causal Transformer**, evaluate job requirements fit, and retrieve direct contact channels.

---

## 🚀 Quick Setup

### 1. Claude Desktop
Add to your `claude_desktop_config.json`:

* **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
* **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "julio-portfolio": {
      "command": "npx",
      "args": ["-y", "byte-od-portfolio-mcp"]
    }
  }
}
```

### 2. Cursor IDE
Go to **Cursor Settings** > **Features** > **MCP Servers** > **Add New MCP Server**:
* **Name**: `julio-portfolio`
* **Type**: `command`
* **Command**: `npx -y byte-od-portfolio-mcp`

### 3. Claude Code / Terminal
```bash
claude mcp add julio-portfolio -- npx -y byte-od-portfolio-mcp
```

### 4. Smithery.ai (1-Click Install)
```bash
npx -y @smithery/cli install byte-od-portfolio-mcp --client claude
```

---

## 🛠️ Available MCP Tools

| Tool | Description |
| :--- | :--- |
| `get_profile` | Returns full canonical biography, seniority level, contact info, and core stack. |
| `get_projects` | Lists flagship projects (CIR-Engine 503M, system_one, mddd-cli, Register-red, MAD, flutter_scene, GitTrack). |
| `inspect_project` | Deep-dive architectural specs, parameter counts, attention mechanisms (RoPE, GQA), and DDP throughput benchmarks (~6,200 tokens/s). |
| `evaluate_job_fit` | Evaluates Julio's candidate fit against a target job title and requirement specs. |
| `get_contact_info` | Returns direct email, LinkedIn, and hiring availability. |

---

## 📦 Publishing to npm

To publish this package under your npm account:
```bash
cd mcp-server
npm login
npm publish --access public
```
*(Once published, any developer in the world can run `npx -y byte-od-portfolio-mcp` instantly!)*

---

## 🌐 Live Portfolio
* **URL**: [https://byte-od.github.io](https://byte-od.github.io)
* **llms.txt**: [https://byte-od.github.io/llms.txt](https://byte-od.github.io/llms.txt)
