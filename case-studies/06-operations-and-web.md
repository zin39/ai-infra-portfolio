# Keeping production running, and building the apps on top

## Systems administration (1.5 years)

- **Monitoring:** Grafana + Prometheus alerting (CPU, memory, disk, network) across production, plus a Telegraf/InfluxDB/Grafana
  stack for the mail servers. Most problems now show up as an alert before a user reports them.
- **Alert noise to signal:** a Python/FastAPI service that parses monitoring and backup-job alert emails (Zabbix, Veeam)
  into structured events and a daily digest, so a failed backup can't hide in an inbox.
- **Database high availability:** planned and rehearsed PostgreSQL streaming replication with a documented manual failover for a
  1.5 TB PostgreSQL/PostGIS database. I rehearsed on test VMs first and kept restarts of the live primary to a minimum.
- **Self-service servers:** Terraform modules for vCenter plus a web portal with an approval workflow and live WebSocket progress.
  The team provisioned 40+ VMs through it in about 4 months. Public code: [terraform-vcenter-vm-portal](https://github.com/zin39/terraform-vcenter-vm-portal)
- **Infrastructure as code on AWS:** VPC, subnets, NAT, EC2 and security groups, deployed by GitHub Actions on every push.
  Public code: [wordpress-aws](https://github.com/zin39/wordpress-aws)
- **Day to day:** email delivery, DNS (BIND), VPS access, SSL, after-hours escalations, and runbooks for the team.

## Web apps and backends

| Project | What it is | Stack |
|---|---|---|
| **WhatsApp + AI messaging platform** | Multi-tenant SaaS: a business connects WhatsApp (official Meta API) to an AI assistant, and staff can take over any conversation from a dashboard. I built the webhook bridge (validation, circuit breaker, secret-free logs, credential rotation). 390+ tests | TypeScript, Fastify, Next.js 15, PostgreSQL, Prisma |
| **Webinar and registration site for a mental-health professional** | Live site with registration, confirmation and reminder emails, an admin panel with CSV export, and analytics with no cookies. Edge caching cut US load times from about 1 s to 54–114 ms. Atomic deploys with rollback | Node.js, Express, Prisma, Docker, Cloudflare |
| **AI video generation pipeline** | Text to narrated video on a GPU: an LLM "director", text-to-speech and a video model, with a Gradio UI and background jobs. I found and removed four GPU-compiler hangs. 113 tests | Python, PyTorch, FastVideo, Gradio |
| **Business group website** | Next.js monorepo with lead capture and a provider-agnostic payments module | Next.js 15, React 19, Drizzle, PostgreSQL |
| **Cloud dashboard** | Web UI for OpenStack | React, Tailwind, Express. Public code: [openstack_dashboard](https://github.com/zin39/openstack_dashboard) |
