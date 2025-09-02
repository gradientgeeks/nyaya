import React, { useRef } from 'react';

interface DocumentUploadProps {
  onFileUpload: (file: File) => void;
  children: React.ReactNode;
}

export const DocumentUpload: React.FC<DocumentUploadProps> = ({ 
  onFileUpload, 
  children 
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      onFileUpload(file);
      // Reset the input value to allow uploading the same file again
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  return (
    <>
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        className="hidden"
        accept=".pdf,.doc,.docx,.txt"
      />
      <div onClick={handleUploadClick}>
        {children}
      </div>
    </>
  );
};