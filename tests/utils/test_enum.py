from nemo.utils.enum import PrettyStrEnum


class ASRModelType(PrettyStrEnum):
    CTC = "ctc"
    RNNT = "rnnt"


class TestPrettyStrEnum:
    def test_incorrect_value(self):
        """Test pretty error message for invalid value"""
        try:
            ASRModelType("incorrect")
        except ValueError as e:
            assert str(e) == "incorrect is not a valid ASRModelType. Possible choices: ctc, rnnt"

    def test_correct_value(self):
        """Test that correct value is accepted"""
        assert ASRModelType("ctc") == ASRModelType.CTC

    def test_str(self):
        """
        Test that str() returns the source value,
        useful for serialization/deserialization and user-friendly logging
        """
        assert str(ASRModelType("ctc")) == "ctc"
