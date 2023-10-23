#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <thread>
#include <memory>

struct StudentId
{
	char str[20];
};

struct StudentMarks
{
	int math;
	int it;
	int lit;
	int pe;
};

struct RecommendLetters
{
	int amount;
};

struct StudentInfo
{
	StudentId id;
	StudentMarks marks;
	RecommendLetters letters;
};

struct StudentInfoRP
{
	StudentInfoRP()
	{
		id = new StudentId();
		marks = new StudentMarks();
		letters = new RecommendLetters();
	}

	~StudentInfoRP()
	{
		delete id;
		delete marks;
		delete letters;
	}

	StudentId* id;
	StudentMarks* marks;
	RecommendLetters* letters;
};

struct StudentInfoSP
{
	std::shared_ptr<StudentId> id = std::make_shared<StudentId>();
	std::shared_ptr<StudentMarks> marks = std::make_shared<StudentMarks>();
	std::shared_ptr<RecommendLetters> letters = std::make_shared<RecommendLetters>();
};

void prnStudentInfo(const std::vector<StudentInfo>& studs)
{
	for (const auto& s : studs)
	{
		std::cout << "id: " << s.id.str << ", marks: " << s.marks.math << " "
			<< s.marks.it << " " << s.marks.lit << " " << s.marks.pe << " "
			<< ", letters: " << s.letters.amount << std::endl;
	}
}

int main()
{
	const int amount = 10 * 1000 * 1000;

	{
		auto start = std::chrono::steady_clock::now();
		std::vector<StudentInfo> studs(amount);
		for (int i = 0; i < amount; ++i)
		{
			StudentInfo& stud = studs[i];
			sprintf_s(stud.id.str, sizeof(stud.id) / sizeof(char), "%03d-%014d", rand() % 100, i);
			stud.marks.math = rand() % 10;
			stud.marks.it = rand() % 10;
			stud.marks.lit = rand() % 10;
			stud.marks.pe = rand() % 10;
			stud.letters.amount = (rand() % 20 == 5) ? rand() % 4 : 0;
		}
		auto end = std::chrono::steady_clock::now();
		const std::chrono::duration<double> createAndInitTime = end - start;

		start = std::chrono::steady_clock::now();
		std::sort(studs.begin(), studs.end(), [](const StudentInfo& a, const StudentInfo& b)
			{
				if (a.marks.math != b.marks.math)
				{
					return a.marks.math < b.marks.math;
				}
				else if (a.marks.it != b.marks.it)
				{
					return a.marks.it < b.marks.it;
				}
				else if (a.marks.lit != b.marks.lit)
				{
					return a.marks.lit < b.marks.lit;
				}
				else if (a.marks.pe != b.marks.pe)
				{
					return a.marks.pe < b.marks.pe;
				}
				else if (a.letters.amount != b.letters.amount)
				{
					return a.letters.amount < b.letters.amount;
				}
				else if (strcmp(a.id.str, b.id.str) != 0)
				{
					return strcmp(a.id.str, b.id.str) < 0;
				}
			});
		end = std::chrono::steady_clock::now();
		std::chrono::duration<double> sortingTime = end - start;
		
		start = std::chrono::steady_clock::now();
		int markSum = 0;
		int letterSum = 0;
		for (const auto& s : studs)
		{
			markSum += s.marks.math;
			markSum += s.marks.it;
			markSum += s.marks.lit;
			markSum += s.marks.pe;
			letterSum += s.letters.amount;
		}
		const float avgMark = (float) markSum / (amount * 4.0f);
		const float avgLetters = (float) letterSum / amount;
		end = std::chrono::steady_clock::now();
		std::chrono::duration<double> readTime = end - start;

		start = std::chrono::steady_clock::now();
		studs.clear();
		end = std::chrono::steady_clock::now();
		std::chrono::duration<double> delTime = end - start;

		std::cout << "Plain Obj,  create and init: " << createAndInitTime.count() << "s, sorting:" << sortingTime.count() 
			<< "s, read: " << readTime.count() << "s, del time: " << delTime.count() << "s, avg mark: " << avgMark << ", avg letters: " << avgLetters << std::endl;
	}

	{
		auto start = std::chrono::steady_clock::now();
		std::vector<StudentInfoRP*> studs;
		studs.reserve(amount);
		for (int i = 0; i < amount; ++i)
		{
			auto stud = new StudentInfoRP();
			sprintf_s(stud->id->str, sizeof(StudentInfo::id) / sizeof(char), "%03d-%014d", rand() % 100, i);
			stud->marks->math = rand() % 10;
			stud->marks->it = rand() % 10;
			stud->marks->lit = rand() % 10;
			stud->marks->pe = rand() % 10;
			stud->letters->amount = (rand() % 20 == 5) ? rand() % 4 : 0;
			studs.push_back(stud);
		}
		auto end = std::chrono::steady_clock::now();
		const std::chrono::duration<double> createAndInitTime = end - start;

		start = std::chrono::steady_clock::now();
		std::sort(studs.begin(), studs.end(), [](const StudentInfoRP* a, const StudentInfoRP* b)
			{
				if (a->marks->math != b->marks->math)
				{
					return a->marks->math < b->marks->math;
				}
				else if (a->marks->it != b->marks->it)
				{
					return a->marks->it < b->marks->it;
				}
				else if (a->marks->lit != b->marks->lit)
				{
					return a->marks->lit < b->marks->lit;
				}
				else if (a->marks->pe != b->marks->pe)
				{
					return a->marks->pe < b->marks->pe;
				}
				else if (a->letters->amount != b->letters->amount)
				{
					return a->letters->amount < b->letters->amount;
				}
				else if (strcmp(a->id->str, b->id->str) != 0)
				{
					return strcmp(a->id->str, b->id->str) < 0;
				}
			});
		end = std::chrono::steady_clock::now();
		const std::chrono::duration<double> sortingTime = end - start;

		start = std::chrono::steady_clock::now();
		int markSum = 0;
		int letterSum = 0;
		for (const auto s : studs)
		{
			markSum += s->marks->math;
			markSum += s->marks->it;
			markSum += s->marks->lit;
			markSum += s->marks->pe;
			letterSum += s->letters->amount;
		}
		const float avgMark = (float)markSum / (amount * 4.0f);
		const float avgLetters = (float)letterSum / amount;
		end = std::chrono::steady_clock::now();
		std::chrono::duration<double> readTime = end - start;

		start = std::chrono::steady_clock::now();
		for (auto s : studs)
			delete s;
		studs.clear();
		end = std::chrono::steady_clock::now();
		std::chrono::duration<double> delTime = end - start;

		std::cout << "Raw Ptr,    create and init: " << createAndInitTime.count() << "s, sorting:" << sortingTime.count()
			<< "s, read: " << readTime.count() << "s, del time: " << delTime.count() << "s, avg mark: " << avgMark << ", avg letters: " << avgLetters << std::endl;
	}

	{
		auto start = std::chrono::steady_clock::now();
		std::vector<std::shared_ptr<StudentInfoRP>> studs;
		studs.reserve(amount);
		for (int i = 0; i < amount; ++i)
		{
			auto stud = std::make_shared<StudentInfoRP>();
			sprintf_s(stud->id->str, sizeof(StudentInfo::id) / sizeof(char), "%03d-%014d", rand() % 100, i);
			stud->marks->math = rand() % 10;
			stud->marks->it = rand() % 10;
			stud->marks->lit = rand() % 10;
			stud->marks->pe = rand() % 10;
			stud->letters->amount = (rand() % 20 == 5) ? rand() % 4 : 0;
			studs.push_back(stud);
		}
		auto end = std::chrono::steady_clock::now();
		const std::chrono::duration<double> createAndInitTime = end - start;

		start = std::chrono::steady_clock::now();
		std::sort(studs.begin(), studs.end(), [](std::shared_ptr<StudentInfoRP> a, std::shared_ptr<StudentInfoRP> b)
			{
				if (a->marks->math != b->marks->math)
				{
					return a->marks->math < b->marks->math;
				}
				else if (a->marks->it != b->marks->it)
				{
					return a->marks->it < b->marks->it;
				}
				else if (a->marks->lit != b->marks->lit)
				{
					return a->marks->lit < b->marks->lit;
				}
				else if (a->marks->pe != b->marks->pe)
				{
					return a->marks->pe < b->marks->pe;
				}
				else if (a->letters->amount != b->letters->amount)
				{
					return a->letters->amount < b->letters->amount;
				}
				else if (strcmp(a->id->str, b->id->str) != 0)
				{
					return strcmp(a->id->str, b->id->str) < 0;
				}
			});
		end = std::chrono::steady_clock::now();
		const std::chrono::duration<double> sortingTime = end - start;

		start = std::chrono::steady_clock::now();
		int markSum = 0;
		int letterSum = 0;
		for (const auto s : studs)
		{
			markSum += s->marks->math;
			markSum += s->marks->it;
			markSum += s->marks->lit;
			markSum += s->marks->pe;
			letterSum += s->letters->amount;
		}
		const float avgMark = (float)markSum / (amount * 4.0f);
		const float avgLetters = (float)letterSum / amount;
		end = std::chrono::steady_clock::now();
		std::chrono::duration<double> readTime = end - start;

		start = std::chrono::steady_clock::now();
		studs.clear();
		end = std::chrono::steady_clock::now();
		std::chrono::duration<double> delTime = end - start;

		std::cout << "Shared Ptr, create and init: " << createAndInitTime.count() << "s, sorting:" << sortingTime.count()
			<< "s, read: " << readTime.count() << "s, del time: " << delTime.count() << "s, avg mark: " << avgMark << ", avg letters: " << avgLetters << std::endl;
	}
}