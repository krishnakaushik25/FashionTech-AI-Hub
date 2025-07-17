import { useState } from "react";
import { Upload, Typography, message } from "antd";
import { InboxOutlined, CloseCircleOutlined } from "@ant-design/icons";

const { Title } = Typography;
const { Dragger } = Upload;

const ImageUpload = ({ label, onImageChange, isDarkMode = false }) => {
  const [preview, setPreview] = useState(null);

  const uploadProps = {
    name: "file",
    multiple: false,
    maxCount: 1,
    accept: "image/*",
    showUploadList: false,
    beforeUpload: (file) => {
      const isImage = file.type.startsWith("image/");
      if (!isImage) {
        message.error("You can only upload image files!");
        return Upload.LIST_IGNORE;
      }

      const isLt10M = file.size / 1024 / 1024 < 10;
      if (!isLt10M) {
        message.error("Image must be smaller than 10MB!");
        return Upload.LIST_IGNORE;
      }

      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
        onImageChange(file);
      };
      reader.readAsDataURL(file);
      return false;
    },
    onDrop: (e) => {
      console.log("Dropped files", e.dataTransfer.files);
    },
  };

  const handleRemove = () => {
    setPreview(null);
    onImageChange(null);
  };

  return (
    <div className="w-full transition-all duration-300 flex flex-col items-center">
      {label && (
        <Title level={5} style={{ marginBottom: "1rem", textAlign: "center" }}>
          {label}
        </Title>
      )}

      {preview ? (
        <div className="relative w-full flex justify-center items-center mt-4">
          <div className="relative">
            <img
              src={preview}
              alt="Preview"
              className="h-[200px] w-[200px] object-contain rounded-lg shadow-md"
              style={{ display: "block" }}
            />

            <CloseCircleOutlined
              onClick={handleRemove}
              style={{
                position: "absolute",
                top: -10,
                right: -10,
                fontSize: 20,
                color: isDarkMode ? "#f87171" : "#ef4444", // red-400 or red-500
                backgroundColor: isDarkMode ? "#1f1f1f" : "#ffffff",
                borderRadius: "50%",
                cursor: "pointer",
                boxShadow: "0 2px 6px rgba(0,0,0,0.2)",
                zIndex: 10,
              }}
            />
          </div>
        </div>
      ) : (
        <Dragger
          {...uploadProps}
          className="w-full max-w-xs p-4"
          style={{
            border: `1px dashed ${isDarkMode ? "#444" : "#d9d9d9"}`,
            borderRadius: 12,
            backgroundColor: isDarkMode ? "#1f1f1f" : "#fafafa",
          }}
        >
          <p className="ant-upload-drag-icon">
            <InboxOutlined
              style={{ color: isDarkMode ? "#38bdf8" : "#1677ff" }}
            />
          </p>
          <p
            className="ant-upload-text"
            style={{ color: isDarkMode ? "#e5e5e5" : "#333" }}
          >
            Click or drag an image here to upload
          </p>
          <p
            className="ant-upload-hint"
            style={{ fontSize: 12, color: isDarkMode ? "#a1a1aa" : "#666" }}
          >
            Image only • Max size: 10MB
          </p>
        </Dragger>
      )}
    </div>
  );
};

export default ImageUpload;
