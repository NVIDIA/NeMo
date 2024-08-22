from dataclasses import dataclass, field
from nemo.lightning import io


class DummyClass(io.IOMixin):
    def __init__(self, a: int, b: int):
        self.a = a
        self.b = b


@dataclass
class BaseDataClass(io.NeedsIOMixin):
    a: int = field(default_factory=lambda: 0)
    b: int = 3

    def lazy_update(self):
        self.mutate_hparam('b', self.b + 2)
        # Will update the value of b set later on making use of a future subclass IOMixin


@dataclass
class OverrideModelDataClass1(BaseDataClass, io.IOMixin):
    a: int = field(default_factory=lambda: 4)
    c: int = 3


@dataclass
class OverrideModelDataClass2(BaseDataClass, io.IOMixin):
    a: int = field(default_factory=lambda: 5)  # override default of a
    # do not define/override b
    c: int = 4  # new variable


class TestIOMixin:
    def test_reinit(self):
        dummy = DummyClass(5, 5)
        copied = io.reinit(dummy)
        assert copied is not dummy
        assert copied.a == dummy.a
        assert copied.b == dummy.b

    def test_two_versions(self):
        v1 = DummyClass(4, 3)
        v2 = DummyClass(3, 2)
        coppied_v1 = io.reinit(v1)
        coppied_v2 = io.reinit(v2)
        assert v1.a != v2.a
        assert v1.a == coppied_v1.a
        assert v2.a == coppied_v2.a

    def test_init(self):
        outputs = []
        for i in range(1001):
            outputs.append(DummyClass(i, i))

        assert len(outputs) == 1001

    def test_dataclasses_two_versions(self):
        _ = OverrideModelDataClass1(b=2)
        v1 = OverrideModelDataClass2(b=4)
        v1.lazy_update()  # the mutate method allows a variable that matches the init arg to be changed and tracked.
        coppied_v1 = io.reinit(v1)  # Simulate loading from a checkpoint
        v2 = OverrideModelDataClass2(a=3, b=1, c=5)
        coppied_v2 = io.reinit(v2)  # Simulate loading from a checkpoint
        assert v1.a != v2.a
        assert v1.a == coppied_v1.a
        assert v2.a == coppied_v2.a
        assert v1.b != v2.b
        assert v1.a == 5
        assert v1.b == 6
        assert v1.c == 4
        assert v2.a == 3
        assert v2.b == 1
        assert v2.c == 5
        assert v1.a == coppied_v1.a
        assert v1.b == coppied_v1.b
        assert v1.c == coppied_v1.c
        assert v2.a == coppied_v2.a
        assert v2.b == coppied_v2.b
        assert v2.c == coppied_v2.c
