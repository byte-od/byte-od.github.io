/**
 * Host Integrity & Anti-Cloning Guard
 * 
 * Verifies domain integrity at runtime. Detects unauthorized mirrors, cloned repositories,
 * or scraper proxies, asserting intellectual property rights under Lei 9.610/98 & DMCA § 512.
 */

const AUTHORIZED_DOMAINS = new Set([
  'byte-od.github.io',
  'localhost',
  '127.0.0.1',
  '0.0.0.0',
]);

const ALLOWED_SUFFIXES = [
  '.local',
  '.internal',
  '.webcontainer.io',
  '.stackblitz.io',
  '.github.dev',
  '.codesandbox.io',
  '.run.app',
];

export function isHostAuthorized(): boolean {
  if (typeof window === 'undefined') return true;
  if (import.meta.env.DEV) return true;

  const hostname = window.location.hostname.toLowerCase();

  if (AUTHORIZED_DOMAINS.has(hostname)) return true;

  for (const suffix of ALLOWED_SUFFIXES) {
    if (hostname.endsWith(suffix)) return true;
  }

  return false;
}

export function logHostIntegrityCheck(): boolean {
  if (typeof window === 'undefined') return true;

  const authorized = isHostAuthorized();
  const currentHost = window.location.hostname;

  if (authorized) {
    console.log(
      '%c[INTEGRITY VERIFIED]%c Running genuine portfolio for Julio Cesar Reis Filho (@byte-od)\n%cCanonical: https://byte-od.github.io | All Rights Reserved.',
      'color: #adff2f; font-weight: bold; font-family: monospace; font-size: 11px;',
      'color: #94a3b8; font-family: monospace; font-size: 11px;',
      'color: #64748b; font-size: 10px; font-family: monospace;'
    );
  } else {
    console.error(
      `%c⚠️ [UNAUTHORIZED CLONE / ESPELHO NÃO AUTORIZADO DETECTADO] ⚠️\n` +
      `Host atual: "${currentHost}" não é um domínio autorizado.\n` +
      `Todo o código, arquitetura Mixture of Scars (MoS) & SCache O(1) e identidade visual\n` +
      `são propriedade intelectual de Julio Cesar Reis Filho (byte-od).\n` +
      `Protegido por Direitos Autorais (Lei nº 9.610/98, Arts. 7º e 28) e DMCA 17 U.S.C. § 512.\n` +
      `Acesse a versão original autorizada em: https://byte-od.github.io`,
      'color: #ff3366; font-weight: bold; font-family: monospace; font-size: 13px; line-height: 1.5;'
    );
  }

  return authorized;
}
