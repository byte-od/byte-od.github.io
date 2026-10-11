import React, { useState } from 'react';
import { ProjectData, getProjects } from '../data/portfolio-data';
import {
  ExternalLink,
  Github,
  Package,
  Check,
  Copy,
} from 'lucide-react';

interface ProjectShowcaseProps {
  lang?: 'en' | 'pt';
}

export const ProjectShowcase: React.FC<ProjectShowcaseProps> = ({ lang = 'en' }) => {
  const [filter, setFilter] = useState<'all' | 'ml_ai' | 'developer_tooling' | 'mobile_graphics'>('all');
  const [activeProjectModal, setActiveProjectModal] = useState<ProjectData | null>(null);
  const [copiedCmd, setCopiedCmd] = useState<string | null>(null);

  const projects = getProjects(lang);
  const filteredProjects = projects.filter((p) => filter === 'all' || p.category === filter);
  const activeProject = activeProjectModal ? (projects.find((p) => p.id === activeProjectModal.id) ?? activeProjectModal) : null;

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedCmd(text);
    setTimeout(() => setCopiedCmd(null), 2000);
  };

  return (
    <div className="space-y-8">
      {/* Filter Tabs (Interactive buttons with click handlers) */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-1 p-1 bg-zinc-900 border border-zinc-800 rounded-lg">
          <button
            type="button"
            onClick={() => setFilter('all')}
            className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors cursor-pointer ${
              filter === 'all' ? 'bg-zinc-800 text-white shadow-xs' : 'text-zinc-400 hover:text-white'
            }`}
          >
            {lang === 'pt' ? 'Todos os Projetos' : 'All Systems'} ({projects.length})
          </button>
          <button
            type="button"
            onClick={() => setFilter('ml_ai')}
            className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors cursor-pointer ${
              filter === 'ml_ai' ? 'bg-zinc-800 text-[#adff2f] shadow-xs' : 'text-zinc-400 hover:text-white'
            }`}
          >
            {lang === 'pt' ? 'ML & LLMs' : 'ML & LLMs'}
          </button>
          <button
            type="button"
            onClick={() => setFilter('developer_tooling')}
            className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors cursor-pointer ${
              filter === 'developer_tooling' ? 'bg-zinc-800 text-[#38bdf8] shadow-xs' : 'text-zinc-400 hover:text-white'
            }`}
          >
            {lang === 'pt' ? 'Ferramentas Dev & CLIs' : 'Developer Tooling & CLIs'}
          </button>
          <button
            type="button"
            onClick={() => setFilter('mobile_graphics')}
            className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors cursor-pointer ${
              filter === 'mobile_graphics' ? 'bg-zinc-800 text-[#c4b5fd] shadow-xs' : 'text-zinc-400 hover:text-white'
            }`}
          >
            {lang === 'pt' ? 'Gráficos & Mobile' : 'Graphics & Mobile'}
          </button>
        </div>

        {/* NPM command snippet quick copy */}
        <div className="flex items-center gap-2 text-xs font-mono bg-zinc-900/80 border border-zinc-800 px-3 py-1.5 rounded-lg text-zinc-300">
          <span className="text-zinc-500">$</span>
          <span>npx mddd-cli --init</span>
          <button
            type="button"
            onClick={() => copyToClipboard('npx mddd-cli --init')}
            className="text-zinc-400 hover:text-[#adff2f] transition-colors ml-1 cursor-pointer"
            title={lang === 'pt' ? 'Copiar comando' : 'Copy command'}
          >
            {copiedCmd === 'npx mddd-cli --init' ? <Check className="w-3.5 h-3.5 text-[#adff2f]" /> : <Copy className="w-3.5 h-3.5" />}
          </button>
        </div>
      </div>

      {/* Projects Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredProjects.map((project) => (
          <div
            key={project.id}
            className="group flex flex-col justify-between p-6 bg-zinc-950/70 border border-zinc-800/80 hover:border-zinc-700 rounded-2xl transition-all duration-200 hover:-translate-y-1 shadow-lg"
          >
            <div>
              {/* Category indicator & Github link */}
              <div className="flex items-center justify-between gap-2 mb-3">
                <span className="text-[11px] font-mono text-zinc-400">
                  {project.badge}
                </span>
                <div className="flex items-center gap-2">
                  {project.packageUrl && (
                    <a
                      href={project.packageUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-zinc-500 hover:text-[#adff2f] transition-colors p-1"
                      title={lang === 'pt' ? 'Pacote NPM' : 'NPM Package'}
                    >
                      <Package className="w-4 h-4" />
                    </a>
                  )}
                  {project.githubUrl ? (
                    <a
                      href={project.githubUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-zinc-500 hover:text-white transition-colors p-1"
                      title={lang === 'pt' ? 'Repositório no GitHub' : 'GitHub Repository'}
                    >
                      <Github className="w-4 h-4" />
                    </a>
                  ) : (
                    <span
                      className="text-zinc-600 px-1 py-0.5 text-[10px] font-mono rounded bg-zinc-900 border border-zinc-800 select-none"
                      title={lang === 'pt' ? 'Repositório Interno' : 'Internal Repository'}
                    >
                      int
                    </span>
                  )}
                </div>
              </div>

              {/* Title */}
              <h4 className="text-lg font-bold text-white group-hover:text-[#adff2f] transition-colors">
                {project.title}
              </h4>

              {/* Specs metadata */}
              {project.specs && (
                <div className="text-[11px] font-mono text-[#38bdf8] mt-1 mb-3">
                  {project.specs}
                </div>
              )}

              {/* Summary */}
              <p className="text-xs text-zinc-400 leading-relaxed mb-4">
                {project.summary}
              </p>

              {/* Metric stats if any */}
              {project.metrics && (
                <div className="grid grid-cols-2 gap-2 my-3 p-2.5 bg-zinc-900/50 rounded-lg border border-zinc-850">
                  {project.metrics.slice(0, 2).map((m, idx) => (
                    <div key={idx} className="flex flex-col">
                      <span className="text-[10px] font-mono text-zinc-500">{m.label}</span>
                      <span className="text-xs font-mono font-bold text-zinc-200 tabular-nums">{m.value}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Footer with Tech Stack and Details button */}
            <div className="pt-4 border-t border-zinc-850 mt-2">
              <div className="flex flex-wrap gap-1.5 mb-3">
                {project.techStack.slice(0, 3).map((tech) => (
                  <span
                    key={tech}
                    className="text-[10px] font-mono text-zinc-400 bg-zinc-900 px-2 py-0.5 rounded border border-zinc-800"
                  >
                    {tech}
                  </span>
                ))}
                {project.techStack.length > 3 && (
                  <span className="text-[10px] font-mono text-zinc-600 px-1 py-0.5">
                    +{project.techStack.length - 3}
                  </span>
                )}
              </div>

              <button
                type="button"
                onClick={() => setActiveProjectModal(project)}
                className="w-full text-xs font-mono py-2 rounded-lg bg-zinc-900 hover:bg-zinc-800 text-zinc-300 hover:text-white border border-zinc-800 transition-colors flex items-center justify-center gap-1.5 cursor-pointer"
              >
                <span>{lang === 'pt' ? 'Ver Especificação Completa' : 'Inspect Specification'}</span>
                <ExternalLink className="w-3 h-3" />
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* Project Detail Modal */}
      {activeProject && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md animate-in fade-in duration-200">
          <div className="relative w-full max-w-2xl bg-zinc-950 border border-zinc-750 rounded-2xl p-6 lg:p-8 shadow-2xl max-h-[90vh] overflow-y-auto">
            {/* Modal Header */}
            <div className="flex items-start justify-between pb-4 border-b border-zinc-800">
              <div>
                <span className="text-xs font-mono text-[#adff2f]">{activeProject.badge}</span>
                <h3 className="text-2xl font-bold text-white mt-1">{activeProject.title}</h3>
                {activeProject.specs && (
                  <p className="text-xs font-mono text-[#38bdf8] mt-0.5">{activeProject.specs}</p>
                )}
              </div>
              <button
                type="button"
                onClick={() => setActiveProjectModal(null)}
                className="text-zinc-400 hover:text-white p-2 rounded-lg hover:bg-zinc-900 transition-colors cursor-pointer"
                title={lang === 'pt' ? 'Fechar' : 'Close'}
              >
                ✕
              </button>
            </div>

            {/* Modal Body */}
            <div className="py-6 space-y-6">
              <div>
                <h5 className="text-xs font-mono uppercase text-zinc-500 tracking-wider mb-2">
                  {lang === 'pt' ? 'Visão Geral' : 'Overview'}
                </h5>
                <p className="text-sm text-zinc-300 leading-relaxed">{activeProject.summary}</p>
              </div>

              {/* Architectural Highlights */}
              <div>
                <h5 className="text-xs font-mono uppercase text-zinc-500 tracking-wider mb-2">
                  {lang === 'pt' ? 'Arquitetura Técnica & Conquistas' : 'Technical Architecture & Achievements'}
                </h5>
                <ul className="space-y-2">
                  {activeProject.details.map((detail, idx) => (
                    <li key={idx} className="text-xs text-zinc-300 flex items-start gap-2 leading-relaxed">
                      <span className="text-[#adff2f] font-mono mt-0.5">▹</span>
                      <span>{detail}</span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Tech Stack Chips */}
              <div>
                <h5 className="text-xs font-mono uppercase text-zinc-500 tracking-wider mb-2">
                  {lang === 'pt' ? 'Stack Tecnológica' : 'Tech Stack'}
                </h5>
                <div className="flex flex-wrap gap-1.5">
                  {activeProject.techStack.map((tech) => (
                    <span
                      key={tech}
                      className="text-xs font-mono text-zinc-300 bg-zinc-900 border border-zinc-800 px-2.5 py-1 rounded"
                    >
                      {tech}
                    </span>
                  ))}
                </div>
              </div>
            </div>

            {/* Modal Footer */}
            <div className="pt-4 border-t border-zinc-850 flex items-center justify-between">
              {activeProject.githubUrl ? (
                <a
                  href={activeProject.githubUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-2 px-4 py-2 bg-white text-black font-mono text-xs font-bold rounded-lg hover:bg-zinc-200 transition-colors"
                >
                  <Github className="w-4 h-4" />
                  <span>{lang === 'pt' ? 'Ver no GitHub' : 'View on GitHub'}</span>
                </a>
              ) : (
                <span className="text-xs font-mono text-zinc-500 italic">
                  {lang === 'pt' ? 'Repositório Interno / Proprietário' : 'Internal / Proprietary Repository'}
                </span>
              )}

              <button
                type="button"
                onClick={() => setActiveProjectModal(null)}
                className="text-xs font-mono text-zinc-400 hover:text-white px-4 py-2 rounded-lg border border-zinc-800 hover:bg-zinc-900 transition-colors cursor-pointer"
              >
                {lang === 'pt' ? 'Fechar' : 'Close'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
