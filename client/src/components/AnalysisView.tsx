import React from 'react';
import type { CaseSummary, AnalysisSection } from '../types';

interface AnalysisViewProps {
  data: CaseSummary;
}

export const AnalysisView: React.FC<AnalysisViewProps> = ({ data }) => {
  const sections: AnalysisSection[] = [
    { title: 'Facts of the Case', key: 'facts', content: data.facts },
    { title: "Petitioner's Arguments", key: 'petitionerArgs', content: data.petitionerArgs },
    { title: "Respondent's Arguments", key: 'respondentArgs', content: data.respondentArgs },
    { title: 'Court Reasoning', key: 'reasoning', content: data.reasoning },
    { title: 'Final Decision', key: 'decision', content: data.decision },
  ];

  return (
    <div className="space-y-4 sm:space-y-6">
      {sections.map(section => {
        if (section.content) {
          return (
            <div key={section.key} className="border-l-4 border-blue-500 pl-4">
              <h3 className="font-bold text-base sm:text-lg mb-2 text-gray-800 dark:text-gray-200">
                {section.title}
              </h3>
              <p className="text-sm sm:text-base text-gray-700 dark:text-gray-300 leading-relaxed">
                {section.content}
              </p>
            </div>
          );
        } else if (data.hasOwnProperty(section.key) && section.content === null) {
          return (
            <div key={section.key} className="border-l-4 border-gray-300 dark:border-gray-600 pl-4">
              <h3 className="font-bold text-base sm:text-lg mb-2 text-gray-800 dark:text-gray-200">
                {section.title}
              </h3>
              <p className="text-sm sm:text-base text-gray-500 dark:text-gray-400 italic">
                This information is not available as the case is still pending.
              </p>
            </div>
          )
        }
        return null;
      })}
    </div>
  );
};