import { Typography } from "antd";

const { Text } = Typography;

const Footer = ({ isDarkMode }) => {
  const background = isDarkMode ? "#1a1a1a" : "#f0f0f0";
  const textColor = isDarkMode ? "#cccccc" : "#333333";

  return (
    <footer
      style={{
        backgroundColor: background,
        padding: "1.5rem 1rem",
        textAlign: "center",
        marginTop: "4rem",
      }}
    >
      <Text style={{ color: textColor, fontSize: 14 }}>
        Developed by{" "}
        <a
          href="https://www.narenderkeswani.com"
          target="_blank"
          rel="noopener noreferrer"
          style={{
            color: textColor,
            textDecoration: "underline",
            fontWeight: 500,
          }}
        >
          Narender Keswani
        </a>{" "}
        • All rights reserved © {new Date().getFullYear()}
      </Text>
    </footer>
  );
};

export default Footer;
