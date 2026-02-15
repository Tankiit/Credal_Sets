import React, { useRef } from 'react';
import html2canvas from 'html2canvas';
import jsPDF from 'jspdf';

const ExportButtons = ({ componentRef }) => {
  const handleExportPNG = async () => {
    if (!componentRef.current) return;
    
    const canvas = await html2canvas(componentRef.current, {
      scale: 2, // Higher quality
      useCORS: true,
      backgroundColor: '#ffffff',
      logging: false,
    });
    
    const link = document.createElement('a');
    link.download = 'comparison-schema.png';
    link.href = canvas.toDataURL('image/png');
    link.click();
  };

  const handleExportPDF = async () => {
    if (!componentRef.current) return;
    
    const canvas = await html2canvas(componentRef.current, {
      scale: 2,
      useCORS: true,
      backgroundColor: '#ffffff',
      logging: false,
    });
    
    const imgData = canvas.toDataURL('image/png');
    const pdf = new jsPDF({
      orientation: 'landscape',
      unit: 'mm',
      format: 'a4',
    });
    
    const imgWidth = 210; // A4 width in mm
    const pageHeight = 297; // A4 height in mm
    const imgHeight = (canvas.height * imgWidth) / canvas.width;
    
    let heightLeft = imgHeight;
    let position = 0;
    
    pdf.addImage(imgData, 'PNG', 0, position, imgWidth, imgHeight);
    heightLeft -= pageHeight;
    
    // Add additional pages if needed
    while (heightLeft >= 0) {
      position = heightLeft - imgHeight;
      pdf.addPage();
      pdf.addImage(imgData, 'PNG', 0, position, imgWidth, imgHeight);
      heightLeft -= pageHeight;
    }
    
    pdf.save('comparison-schema.pdf');
  };

  return (
    <div className="flex gap-4 p-4 bg-gray-100 rounded-lg">
      <button
        onClick={handleExportPNG}
        className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition"
      >
        Export as PNG
      </button>
      <button
        onClick={handleExportPDF}
        className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600 transition"
      >
        Export as PDF
      </button>
    </div>
  );
};

// In your main component:
const ComparisonSchema = () => {
  const componentRef = useRef(null);
  
  return (
    <div>
      <ExportButtons componentRef={componentRef} />
      <div ref={componentRef}>
        {/* Your existing component JSX */}
      </div>
    </div>
  );
};
