import { useState, useEffect, useRef } from "react";
import axios from "axios";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import { Select } from 'antd'; 
import {
  Layout,
  ConfigProvider,
  theme,
  Button,
  Typography,
  Space,
  Switch,
  Input,
  Row,
  Col,
  Divider,
} from "antd";
import {
  BulbOutlined,
  BulbFilled,
  GithubOutlined,
  LinkedinOutlined,
} from "@ant-design/icons";

import ImageUpload from "./components/ImageUpload";
import Footer from './components/Footer'; 
const { Header, Content} = Layout;
const { Title, Text } = Typography;

function App() {
  const [personImage, setPersonImage] = useState(null);
  const [clothImage, setClothImage] = useState(null);
  const [instructions, setInstructions] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [isDarkMode, setIsDarkMode] = useState(() => {
    const savedMode = localStorage.getItem("darkMode");
    return savedMode ? JSON.parse(savedMode) : false;
  });

  const [modelType, setModelType] = useState("");
  const [gender, setGender] = useState("");
  const [garmentType, setGarmentType] = useState("");
  const [style, setStyle] = useState("");

  const { Option } = Select;

  const resultRef = useRef(null);

  const { defaultAlgorithm, darkAlgorithm } = theme;

  useEffect(() => {
    localStorage.setItem("darkMode", JSON.stringify(isDarkMode));
  }, [isDarkMode]);

  useEffect(() => {
    if (result && resultRef.current) {
      resultRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [result]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!personImage || !clothImage) {
      toast.error("Please upload both person and cloth images");
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append("person_image", personImage);
    formData.append("cloth_image", clothImage);
    formData.append("instructions", instructions);
    
    // Add dropdown values to form data
    formData.append("model_type", modelType || "");
    formData.append("gender", gender || "");
    formData.append("garment_type", garmentType || "");
    formData.append("style", style || "");

    try {
      const response = await axios.post("http://localhost:8000/api/try-on", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      const newResult = {
        id: Date.now(),
        resultImage: response.data.image,
        text: response.data.text,
        timestamp: new Date().toLocaleString(),
      };

      setResult(newResult);
      setHistory((prev) => [newResult, ...prev]);
      toast.success("Virtual try-on completed successfully!");
    } catch (error) {
      toast.error(
        error.response?.data?.message || "An error occurred during processing"
      );
    } finally {
      setLoading(false);
    }
  };

  const bgColor = isDarkMode ? "#0f0f0f" : "#f9fafb";
  const cardColor = isDarkMode ? "#1c1c1c" : "#ffffff";
  const textColor = isDarkMode ? "#e4e4e4" : "#111827";
  const subText = isDarkMode ? "#9ca3af" : "#4b5563";

  return (
    <ConfigProvider
      theme={{
        algorithm: isDarkMode ? darkAlgorithm : defaultAlgorithm,
        token: {
          colorPrimary: "#0ea5e9",
          borderRadius: 10,
        },
      }}
    >
      <Layout style={{ minHeight: "100vh", background: bgColor }}>
        <Header
          style={{
            background: "transparent",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            padding: "1.5rem 2rem",
          }}
        >
          <Title level={3} style={{ margin: 0, color: textColor }}>
            👗 Virtual Try-On
          </Title>
          <Switch
            checked={isDarkMode}
            onChange={setIsDarkMode}
            checkedChildren={<BulbFilled />}
            unCheckedChildren={<BulbOutlined />}
          />
        </Header>
        <Content style={{ padding: "2rem 1rem" }}>
          <div className="max-w-5xl mx-auto">
            <Title
              level={1}
              className="text-center"
              style={{ color: textColor, marginBottom: 40 }}
            >
              Try-On Clothes in Seconds
            </Title>

            <form onSubmit={handleSubmit}>
              <Row gutter={[24, 24]}>
                {/* Model Section */}
                <Col xs={24} md={12}>
                  <div
                    style={{
                      background: cardColor,
                      padding: 24,
                      borderRadius: 12,
                    }}
                  >
                    <Title
                      level={4}
                      style={{ color: textColor, marginBottom: 16 }}
                    >
                      Model Image
                    </Title>

                    <ImageUpload
                      label="Upload Model Image"
                      onImageChange={setPersonImage}
                      isDarkMode={isDarkMode}
                    />

                    <div className="mt-6 space-y-4">
                      {/* Model Type */}
                      <div>
                        <Text style={{ color: subText }}>Model Type</Text>
                        <Select
                          placeholder="Select model type"
                          style={{ width: "100%", marginTop: 4 }}
                          value={modelType}
                          onChange={setModelType}
                        >
                          <Option value="top">Top Half</Option>
                          <Option value="bottom">Bottom Half</Option>
                          <Option value="full">Full Body</Option>
                        </Select>
                      </div>

                      {/* Gender */}
                      <div>
                        <Text style={{ color: subText }}>Gender</Text>
                        <Select
                          placeholder="Select gender"
                          style={{ width: "100%", marginTop: 4 }}
                          value={gender}
                          onChange={setGender}
                        >
                          <Option value="male">Male</Option>
                          <Option value="female">Female</Option>
                          <Option value="unisex">Unisex</Option>
                        </Select>
                      </div>
                    </div>
                  </div>
                </Col>

                {/* Garment Section */}
                <Col xs={24} md={12}>
                  <div
                    style={{
                      background: cardColor,
                      padding: 24,
                      borderRadius: 12,
                    }}
                  >
                    <Title
                      level={4}
                      style={{ color: textColor, marginBottom: 16 }}
                    >
                      Garment Image
                    </Title>

                    <ImageUpload
                      label="Upload Cloth Image"
                      onImageChange={setClothImage}
                      isDarkMode={isDarkMode}
                    />

                    <div className="mt-6 space-y-4">
                      {/* Garment Type */}
                      <div>
                        <Text style={{ color: subText }}>Garment Type</Text>
                        <Select
                          placeholder="Select garment type"
                          style={{ width: "100%", marginTop: 4 }}
                          value={garmentType}
                          onChange={setGarmentType}
                        >
                          <Option value="shirt">Shirt</Option>
                          <Option value="pants">Pants</Option>
                          <Option value="jacket">Jacket</Option>
                          <Option value="dress">Dress</Option>
                          <Option value="tshirt">T-shirt</Option>
                        </Select>
                      </div>

                      {/* Style */}
                      <div>
                        <Text style={{ color: subText }}>Style</Text>
                        <Select
                          placeholder="Select style"
                          style={{ width: "100%", marginTop: 4 }}
                          value={style}
                          onChange={setStyle}
                        >
                          <Option value="casual">Casual</Option>
                          <Option value="formal">Formal</Option>
                          <Option value="streetwear">Streetwear</Option>
                          <Option value="traditional">Traditional</Option>
                          <Option value="sports">Sportswear</Option>
                        </Select>
                      </div>
                    </div>
                  </div>
                </Col>
              </Row>

              {/* Instructions */}
              <div style={{ marginTop: "2.5rem" }}>
                <Title
                  level={5}
                  style={{ color: textColor, marginBottom: "0.5rem" }}
                >
                  Special Instructions
                </Title>
                <Input.TextArea
                  value={instructions}
                  onChange={(e) => setInstructions(e.target.value)}
                  rows={4}
                  placeholder="e.g. Fit for walking pose, crop top, side view preferred..."
                  style={{
                    borderRadius: 10,
                    padding: "1rem",
                    fontSize: "1rem",
                    backgroundColor: isDarkMode ? "#1f1f1f" : "#ffffff",
                    color: textColor,
                    borderColor: isDarkMode ? "#333" : "#d1d5db",
                  }}
                />
              </div>

              {/* Submit Button */}
              <div
                style={{
                  display: "flex",
                  justifyContent: "center",
                  marginTop: "3rem",
                }}
              >
                <Button
                  type="primary"
                  size="large"
                  htmlType="submit"
                  loading={loading}
                  style={{
                    height: 48,
                    width: 200,
                    fontSize: 16,
                    borderRadius: 8,
                  }}
                >
                  {loading ? "Processing..." : "Try On"}
                </Button>
              </div>
            </form>

            {result && (
              <div ref={resultRef} className="mt-20">
                <Divider />
                <Title
                  level={3}
                  style={{
                    color: textColor,
                    textAlign: "center",
                    marginBottom: 32,
                  }}
                >
                  Your Try-On Result
                </Title>
                <div className="flex justify-center">
                  <img
                    src={result.resultImage}
                    alt="Try-On Result"
                    style={{
                      borderRadius: 16,
                      boxShadow: "0 10px 40px rgba(0,0,0,0.3)",
                      maxHeight: 480,
                    }}
                  />
                </div>
                <Text
                  style={{
                    display: "block",
                    textAlign: "center",
                    marginTop: 16,
                    color: isDarkMode ? "#ffffff" : "#000000",
                    fontSize: "1.25rem",
                    fontWeight: "600"
                  }}
                >
                  {result.text}
                </Text>
              </div>
            )}

            {history.length > 0 && (
              <div className="mt-24">
                <Divider />
                <Title level={3} style={{ color: textColor, marginBottom: 32 }}>
                  Previous Results
                </Title>
                <Row gutter={[24, 24]}>
                  {history.map((item) => (
                    <Col xs={24} sm={12} md={8} key={item.id}>
                      <div
                        style={{
                          background: cardColor,
                          padding: 16,
                          borderRadius: 12,
                        }}
                      >
                        <img
                          src={item.resultImage}
                          alt="Previous"
                          style={{
                            width: "100%",
                            borderRadius: 10,
                            marginBottom: 12,
                          }}
                        />
                        <Text
                          style={{
                            display: "block",
                            color: isDarkMode ? "#ffffff" : "#000000",
                            fontSize: "1.25rem",
                            fontWeight: "600",
                            marginBottom: 4,
                          }}
                        >
                          {item.text}
                        </Text>
                        <Text
                          style={{
                            color: isDarkMode ? "#777" : "#666",
                            fontSize: 12,
                          }}
                        >
                          {item.timestamp}
                        </Text>
                      </div>
                    </Col>
                  ))}
                </Row>
              </div>
            )}
          </div>
        </Content>

        <Footer isDarkMode={isDarkMode} />

        <ToastContainer theme={isDarkMode ? "dark" : "light"} />
      </Layout>
    </ConfigProvider>
  );
}

export default App;
